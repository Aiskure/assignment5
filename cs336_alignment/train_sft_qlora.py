import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # logging / eval / checkpoint
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N steps (0=disabled)")
    parser.add_argument("--save_final", action=argparse.BooleanOptionalAction, default=True)

    # resume (optional manual override; by default auto-detects latest checkpoint)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint dir. If omitted, auto-detects latest in output_dir.")

    # wandb
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
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


def compute_loss(model, batch):
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
def evaluate(model, dataloader):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        loss = compute_loss(model, batch)
        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir):
    """Scan output_dir/checkpoints/ and return the path with the highest step number."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    steps = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("step_"):
            try:
                steps.append(int(name.split("_")[1]))
            except ValueError:
                continue
    if not steps:
        return None
    return os.path.join(ckpt_dir, f"step_{max(steps)}")


def _save_adapter(model, tokenizer, adapter_dir, accelerator=None):
    """Save LoRA adapter weights (primary artifact)."""
    if accelerator:
        accelerator.wait_for_everyone()
    if accelerator is None or accelerator.is_main_process:
        os.makedirs(adapter_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model) if accelerator else model
        unwrapped.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[CHECKPOINT] Adapter saved to {adapter_dir}")


def _save_full_checkpoint(model, tokenizer, optimizer, adapter_dir, accelerator,
                          global_step, epoch, batch_idx, wandb_run_id=None):
    """Save adapter + optimizer + RNG states + training state (for seamless resume)."""
    _save_adapter(model, tokenizer, adapter_dir, accelerator)
    if accelerator.is_main_process:
        # optimizer state
        torch.save(optimizer.state_dict(), os.path.join(adapter_dir, "optimizer.pt"))
        # RNG states
        rng_state = {
            "python": random.getstate(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, os.path.join(adapter_dir, "rng_state.pt"))
        # training state
        state = {
            "global_step": global_step,
            "epoch": epoch,
            "batch_idx": batch_idx,
        }
        if wandb_run_id:
            state["wandb_run_id"] = wandb_run_id
        with open(os.path.join(adapter_dir, "training_state.json"), "w") as f:
            json.dump(state, f)


def _save_merged(model, tokenizer, final_dir, accelerator=None):
    """Merge LoRA adapter into base model and save full weights (for evaluation compatibility)."""
    if accelerator:
        accelerator.wait_for_everyone()
    if accelerator is None or accelerator.is_main_process:
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model) if accelerator else model
        merged = unwrapped.merge_and_unload()
        merged.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"[CHECKPOINT] Merged model saved to {final_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # accelerator (no FSDP, single GPU with gradient accumulation)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Auto-detect resume checkpoint ---
    resume_ckpt = args.resume_from or _find_latest_checkpoint(args.output_dir)
    saved_training_state = None
    if resume_ckpt:
        state_path = os.path.join(resume_ckpt, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                saved_training_state = json.load(f)

    # --- wandb (auto-resume if run_id found in checkpoint) ---
    wandb_run = None
    wandb_run_id = None
    if args.wandb_project and HAS_WANDB and accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config=vars(args),
            dir=args.output_dir,
        )
        # Resume the same wandb run if we have a saved run_id
        if saved_training_state and saved_training_state.get("wandb_run_id"):
            wandb_kwargs["id"] = saved_training_state["wandb_run_id"]
            wandb_kwargs["resume"] = "must"
        wandb_run = wandb.init(**wandb_kwargs)
        wandb_run_id = wandb_run.id

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # model — 4-bit quantized
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # prepare for kbit training (enables gradient checkpointing on quantized model)
    model = prepare_model_for_kbit_training(model)

    # LoRA adapter
    if resume_ckpt:
        model = PeftModel.from_pretrained(model, resume_ckpt, is_trainable=True)
        if accelerator.is_main_process:
            print(f"[RESUME] Loaded adapter from {resume_ckpt}")
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

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

    # optimizer — only trainable (LoRA) parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    # prepare with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader,
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # training state
    global_step = 0
    resume_batch_idx = 0
    resume_epoch = 0

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    min_lr = args.learning_rate * 0.1

    # --- Restore training state on resume ---
    if resume_ckpt:
        if saved_training_state:
            global_step = saved_training_state["global_step"]
            resume_epoch = saved_training_state["epoch"]
            resume_batch_idx = saved_training_state["batch_idx"] + 1
        else:
            # Infer from directory name (legacy checkpoints without training_state.json)
            dirname = os.path.basename(resume_ckpt.rstrip("/"))
            if dirname.startswith("step_"):
                global_step = int(dirname.split("_")[1])
                resume_batch_idx = global_step * args.gradient_accumulation_steps
            if accelerator.is_main_process:
                print(f"[RESUME] No training_state.json, inferred global_step={global_step}")

        # Restore optimizer state
        opt_path = os.path.join(resume_ckpt, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=accelerator.device))
            if accelerator.is_main_process:
                print(f"[RESUME] Loaded optimizer state")
        else:
            if accelerator.is_main_process:
                print("[RESUME] No optimizer.pt found, optimizer starts fresh")

        # Restore RNG states
        rng_path = os.path.join(resume_ckpt, "rng_state.pt")
        if os.path.exists(rng_path):
            rng_state = torch.load(rng_path, map_location="cpu")
            random.setstate(rng_state["python"])
            torch.random.set_rng_state(rng_state["torch"])
            if rng_state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            if accelerator.is_main_process:
                print("[RESUME] Restored RNG states")

        if accelerator.is_main_process:
            print(f"[RESUME] global_step={global_step}, epoch={resume_epoch}, "
                  f"batch_idx={resume_batch_idx}, remaining={total_steps - global_step}")

    # --- Training loop ---
    model.train()

    pbar = None
    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, initial=global_step, desc="Training")

    for epoch in range(resume_epoch, args.num_epochs):
        accumulated_loss = 0.0
        skip_batches = resume_batch_idx if epoch == resume_epoch else 0
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx < skip_batches:
                continue
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
                        val_loss = evaluate(model, val_dataloader)
                        if accelerator.is_main_process:
                            print(f"epoch={epoch} step={global_step} val_loss={val_loss:.4f}")
                            if wandb_run:
                                wandb_run.log({"eval/loss": val_loss}, step=global_step)

                    if args.save_every > 0 and global_step % args.save_every == 0:
                        ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}")
                        _save_full_checkpoint(
                            model, tokenizer, optimizer, ckpt_dir, accelerator,
                            global_step, epoch, batch_idx, wandb_run_id,
                        )

    if pbar is not None:
        pbar.close()

    # save final checkpoint
    if args.save_final:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        _save_adapter(model, tokenizer, adapter_dir, accelerator)

        final_dir = os.path.join(args.output_dir, "final_checkpoint")
        _save_merged(model, tokenizer, final_dir, accelerator)

    if wandb_run:
        wandb_run.finish()

    if accelerator.is_main_process:
        print(f"[DONE] Training finished. Total steps: {global_step}")


if __name__ == "__main__":
    main()
