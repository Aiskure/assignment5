import argparse
import math
import os
import random

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    HAS_WANDB = True
except ModuleNotFoundError:
    wandb = None
    HAS_WANDB = False

from cs336_alignment.data_loading import PackedSFTDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    # data
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # logging / eval / checkpoint
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N steps (0=disabled)")
    parser.add_argument("--save_final", action=argparse.BooleanOptionalAction, default=True)

    # wandb
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def get_cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_dataloader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def compute_loss(model, batch, device=None):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    logits = model(input_ids=input_ids).logits
    # logits: (B, T, V), labels: (B, T)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )
    return loss


@torch.no_grad()
def evaluate(model, dataloader, device, accelerator):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        loss = compute_loss(model, batch)
        total_loss += loss.item()
        total_batches += 1

    model.train()
    # average across processes
    avg_loss = total_loss / max(total_batches, 1)
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    avg_loss_tensor = accelerator.reduce(avg_loss_tensor, reduction="mean")
    return avg_loss_tensor.item()


def _save_checkpoint_fsdp(accelerator, model, tokenizer, checkpoint_dir):
    """Save full model weights that can be loaded by from_pretrained()."""
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        unwrapped.save_pretrained(
            checkpoint_dir,
            is_main_process=True,
            state_dict=accelerator.get_state_dict(model),
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"[CHECKPOINT] Saved to {checkpoint_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # accelerator with FSDP (configured via accelerate config / env vars)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    device = accelerator.device

    # wandb — only on main process
    wandb_run = None
    if args.wandb_project and HAS_WANDB and accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir,
        )

    # tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # datasets
    train_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=args.train_dataset_path,
        seq_length=args.seq_length,
        shuffle=args.shuffle,
    )

    val_dataset = None
    if args.val_dataset_path is not None:
        val_dataset = PackedSFTDataset(
            tokenizer=tokenizer,
            dataset_path=args.val_dataset_path,
            seq_length=args.seq_length,
            shuffle=False,
        )

    # dataloaders
    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )

    # optimizer — decay only 2D+ params (weights), not 1D (biases, layernorms)
    params_to_decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    params_no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": params_to_decay, "weight_decay": args.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        foreach=False,
    )

    # prepare with accelerator (handles FSDP wrapping)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader,
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # training loop
    global_step = 0
    model.train()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    min_lr = args.learning_rate * 0.1

    pbar = None
    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, desc="Training")

    for epoch in range(args.num_epochs):
        accumulated_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = compute_loss(model, batch)
                accelerator.backward(loss)
                accumulated_loss += loss.item() / args.gradient_accumulation_steps

                if accelerator.sync_gradients:
                    # LR scheduling
                    if args.lr_scheduler == "cosine":
                        lr = get_cosine_lr(global_step, args.learning_rate, min_lr, warmup_steps, total_steps)
                    else:
                        lr = args.learning_rate
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                    if args.grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(epoch=epoch, loss=f"{accumulated_loss:.4f}", lr=f"{lr:.2e}")

                    if global_step % args.log_every == 0 and accelerator.is_main_process:
                        print(f"epoch={epoch} step={global_step} train_loss={accumulated_loss:.4f} lr={lr:.2e}")
                        if wandb_run:
                            wandb_run.log({"train/loss": accumulated_loss, "train/lr": lr}, step=global_step)

                    accumulated_loss = 0.0

                    if val_dataloader is not None and args.eval_every > 0 and global_step % args.eval_every == 0:
                        val_loss = evaluate(model, val_dataloader, device, accelerator)
                        if accelerator.is_main_process:
                            print(f"epoch={epoch} step={global_step} val_loss={val_loss:.4f}")
                            if wandb_run:
                                wandb_run.log({"eval/loss": val_loss}, step=global_step)

                    if args.save_every > 0 and global_step % args.save_every == 0:
                        ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}")
                        _save_checkpoint_fsdp(accelerator, model, tokenizer, ckpt_dir)

    if pbar is not None:
        pbar.close()

    # save final checkpoint
    if args.save_final:
        final_dir = os.path.join(args.output_dir, "final_checkpoint")
        _save_checkpoint_fsdp(accelerator, model, tokenizer, final_dir)

    if wandb_run:
        wandb_run.finish()

    if accelerator.is_main_process:
        print(f"[DONE] Training finished. Total steps: {global_step}")


if __name__ == "__main__":
    main()
