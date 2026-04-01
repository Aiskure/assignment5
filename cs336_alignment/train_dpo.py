from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.utils import (
    get_response_log_probs,
    compute_per_instance_dpo_loss,
    ALPACA_TEMPLATE,
)



def _find_latest_checkpoint(output_dir):
    """扫描 output_dir/ 下的 step_* 目录，返回最新 checkpoint 路径。"""
    if not os.path.isdir(output_dir):
        return None
    steps = []
    for name in os.listdir(output_dir):
        if name.startswith("step_"):
            try:
                steps.append(int(name.split("_")[1]))
            except ValueError:
                continue
    if not steps:
        return None
    return os.path.join(output_dir, f"step_{max(steps)}")


def _save_full_checkpoint(model, tokenizer, optimizer, save_dir,
                          global_step, epoch, sample_idx, best_val_acc,
                          metrics_history, wandb_run_id=None):
    """保存完整 checkpoint：model + tokenizer + optimizer + RNG + training state。"""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "sample_idx": sample_idx,
        "best_val_acc": best_val_acc,
        "metrics_history": metrics_history,
        "wandb_run_id": wandb_run_id,
        "rng_python": random.getstate(),
        "rng_torch": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, os.path.join(save_dir, "training_state.pt"))
    with open(os.path.join(save_dir, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)


def preprocess_val_data(val_data, tokenizer, max_length=512):
    """提前 tokenize val 数据，避免每次 eval 重复 tokenize。"""
    cached = []
    for item in val_data:
        chosen_text = ALPACA_TEMPLATE.format(
            instruction=item["instruction"], response=item["chosen_response"]
        ) + tokenizer.eos_token
        rejected_text = ALPACA_TEMPLATE.format(
            instruction=item["instruction"], response=item["rejected_response"]
        ) + tokenizer.eos_token
        chosen_ids = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        rejected_ids = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        cached.append((chosen_ids, rejected_ids))
    return cached


def compute_val_accuracy(model, model_ref, val_cached):
    """计算 validation set 上的 implicit reward model classification accuracy。

    val_cached: preprocess_val_data 返回的 list[(chosen_ids, rejected_ids)]

    当 chosen 的 implicit reward 高于 rejected 时算分类正确：
    即 log π_θ(y_w|x) - log π_ref(y_w|x) > log π_θ(y_l|x) - log π_ref(y_l|x)
    （β 是公因子，不影响大小比较，故不需要传入）
    """
    model.eval()
    lm_device = next(model.parameters()).device
    ref_device = next(model_ref.parameters()).device
    correct = 0
    with torch.no_grad():
        for chosen_ids, rejected_ids in val_cached:
            # π_θ log-probs
            lm_chosen = get_response_log_probs(
                model, chosen_ids[:, :-1].to(lm_device), chosen_ids[:, 1:].to(lm_device)
            )["log_probs"].sum()
            lm_rejected = get_response_log_probs(
                model, rejected_ids[:, :-1].to(lm_device), rejected_ids[:, 1:].to(lm_device)
            )["log_probs"].sum()

            # π_ref log-probs
            ref_chosen = get_response_log_probs(
                model_ref, chosen_ids[:, :-1].to(ref_device), chosen_ids[:, 1:].to(ref_device)
            )["log_probs"].sum()
            ref_rejected = get_response_log_probs(
                model_ref, rejected_ids[:, :-1].to(ref_device), rejected_ids[:, 1:].to(ref_device)
            )["log_probs"].sum()

            # implicit reward: log π_θ(y|x) - log π_ref(y|x)
            reward_chosen = lm_chosen - ref_chosen.to(lm_device)
            reward_rejected = lm_rejected - ref_rejected.to(lm_device)
            if reward_chosen > reward_rejected:
                correct += 1

    model.train()
    return correct / len(val_cached)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="SFT 模型路径（已 merge 的完整模型，或 base + adapter）")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="QLoRA adapter 路径（如果模型未 merge）")
    parser.add_argument("--train_data_path", type=str, default="data/hh/processed_train.jsonl")
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/dpo/run1")
    parser.add_argument("--batch_size", type=int, default=64, help="有效 batch size（梯度累积）")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=100, help="每多少步做一次 validation")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512, help="tokenize 截断长度（训练+val 共用）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--ref_device", type=str, default="cuda:1")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="手动指定 resume checkpoint 路径（默认自动检测 output_dir 下最新）")
    parser.add_argument("--wandb_project", type=str, default="cs336-a5-sft",
                        help="W&B project name (None to disable)")
    parser.add_argument("--wandb_offline", action="store_true", default=True)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 0. 检测 Resume（在任何随机操作之前恢复 RNG）----
    resume_ckpt = args.resume_from or _find_latest_checkpoint(args.output_dir)
    resume_epoch = 0
    resume_sample_idx = 0
    global_step = 0
    best_val_acc = 0.0
    metrics_history = []
    wandb_run_id = None
    has_valid_resume = (resume_ckpt
                        and os.path.exists(os.path.join(resume_ckpt, "training_state.pt")))

    if has_valid_resume:
        print(f"\n==> Resuming from {resume_ckpt}")
        state = torch.load(os.path.join(resume_ckpt, "training_state.pt"),
                           map_location="cpu")
        # 先恢复 RNG，确保后续所有 shuffle 与中断前一致
        random.setstate(state["rng_python"])
        torch.random.set_rng_state(state["rng_torch"])
        if state.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        global_step = state["global_step"]
        resume_epoch = state["epoch"]
        resume_sample_idx = state.get("sample_idx", 0)
        best_val_acc = state["best_val_acc"]
        metrics_history = state.get("metrics_history", [])
        wandb_run_id = state.get("wandb_run_id")
        print(f"    global_step={global_step}, epoch={resume_epoch}, "
              f"sample_idx={resume_sample_idx}, best_val_acc={best_val_acc:.4f}")
    elif resume_ckpt:
        print(f"Warning: checkpoint {resume_ckpt} found but no training_state.pt, starting fresh")

    # 保存 config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {args.output_dir}/config.json")

    # ---- 1. 加载数据 ----
    from cs336_alignment.data_load_hh import load_hh_data, filter_to_single_turn, parse_conversation

    if args.train_data_path.endswith(".jsonl"):
        with open(args.train_data_path) as f:
            all_data = [json.loads(line) for line in f]
    else:
        files = [
            "data/hh/harmless-base.jsonl",
            "data/hh/helpful-base.jsonl",
            "data/hh/helpful-online.jsonl",
            "data/hh/helpful-rejection-sampled.jsonl",
        ]
        all_data = []
        for fp in files:
            items = load_hh_data(fp)
            for item in items:
                item["source"] = fp.split("/")[-1].replace(".jsonl", "")
            all_data.extend(items)
        all_data = filter_to_single_turn(all_data)
        all_data = parse_conversation(all_data)

    # 拆分 train/val（RNG 已恢复，shuffle 顺序与中断前一致）
    random.shuffle(all_data)
    if args.val_data_path:
        with open(args.val_data_path) as f:
            val_data = [json.loads(line) for line in f]
        train_data = all_data
    else:
        val_data = all_data[:args.val_size]
        train_data = all_data[args.val_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # ---- 数据统计 ----
    from collections import Counter
    source_counts = Counter(item.get("source", "unknown") for item in train_data)
    chosen_lens = [len(item["chosen_response"]) for item in train_data]
    rejected_lens = [len(item["rejected_response"]) for item in train_data]
    instruction_lens = [len(item["instruction"]) for item in train_data]
    print(f"\n--- Data Statistics (char-level) ---")
    print(f"Sources: {dict(source_counts)}")
    print(f"Instruction len: avg={sum(instruction_lens)/len(instruction_lens):.0f}, "
          f"max={max(instruction_lens)}, min={min(instruction_lens)}")
    print(f"Chosen len:      avg={sum(chosen_lens)/len(chosen_lens):.0f}, "
          f"max={max(chosen_lens)}, min={min(chosen_lens)}")
    print(f"Rejected len:    avg={sum(rejected_lens)/len(rejected_lens):.0f}, "
          f"max={max(rejected_lens)}, min={min(rejected_lens)}")
    len_diffs = [c - r for c, r in zip(chosen_lens, rejected_lens)]
    print(f"Chosen - Rejected: avg={sum(len_diffs)/len(len_diffs):.0f}, "
          f"max={max(len_diffs)}, min={min(len_diffs)}")
    print(f"-----------------------------------\n")

    # π_θ 的加载路径：resume 时从 checkpoint 加载，否则从初始 SFT 模型加载
    model_load_path = resume_ckpt if has_valid_resume else args.model_name_or_path

    # ---- 2. 加载模型 ----
    print(f"Loading tokenizer from {model_load_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 预处理 val 数据
    print("Preprocessing val data (tokenize once)...")
    val_cached = preprocess_val_data(val_data, tokenizer, max_length=args.max_length)

    print(f"Loading π_θ from {model_load_path} on {args.train_device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path, torch_dtype=torch.bfloat16
    ).to(args.train_device)

    # 仅首次训练时需要 merge adapter（resume 时 checkpoint 已包含完整权重）
    if args.adapter_path and not has_valid_resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
        model = model.merge_and_unload()
        print(f"Merged adapter from {args.adapter_path}")

    # π_ref 始终从初始 SFT 模型加载
    print(f"Loading π_ref from {args.model_name_or_path} on {args.ref_device}")
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16
    ).to(args.ref_device)

    if args.adapter_path:
        from peft import PeftModel as PeftModel2
        model_ref_peft = PeftModel2.from_pretrained(model_ref, args.adapter_path, is_trainable=False)
        model_ref = model_ref_peft.merge_and_unload()
        print(f"Merged adapter into ref model")

    model.config.use_cache = False
    model_ref.config.use_cache = False
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad = False

    # ---- 3. 优化器 ----
    optimizer = torch.optim.RMSprop(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    # ---- 3.5 Resume：恢复 optimizer state（RNG/训练进度已在 section 0 恢复）----
    if has_valid_resume:
        optimizer.load_state_dict(state["optimizer"])
        print(f"    Optimizer state restored from {resume_ckpt}")

    # ---- W&B ----
    wandb_run = None
    if args.wandb_project and HAS_WANDB:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        wandb_kwargs = dict(
            project=args.wandb_project,
            group="dpo",
            name=args.wandb_run_name or os.path.basename(args.output_dir),
            config=vars(args),
            dir=str(args.output_dir),
        )
        if wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"
        wandb_run = wandb.init(**wandb_kwargs)
        wandb_run_id = wandb_run.id
        print(f"[INFO] W&B initialized: project={args.wandb_project} group=dpo offline={args.wandb_offline}")
    elif args.wandb_project and not HAS_WANDB:
        print("[WARN] wandb not installed, skipping W&B logging")

    # ---- 4. 训练循环 ----
    effective_batch_size = args.batch_size
    gradient_accumulation_steps = effective_batch_size  # 每条单独算 loss，累积到 effective_batch_size
    total_steps = math.ceil(len(train_data) * args.num_epochs / gradient_accumulation_steps)
    print(f"\n{'='*60}")
    print(f"Total training samples: {len(train_data)}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Epochs: {args.num_epochs}")
    if global_step > 0:
        print(f"Resuming from step {global_step}/{total_steps}")
    print(f"{'='*60}\n")

    running_loss = 0.0
    train_start_time = time.time()

    # Step 0 validation baseline（仅首次训练）
    if global_step == 0:
        val_acc = compute_val_accuracy(model, model_ref, val_cached)
        print(f"[Step 0] val_accuracy={val_acc:.4f} (baseline before training)")
        metrics_history.append({"step": 0, "val_accuracy": val_acc})
        best_val_acc = val_acc
        if wandb_run:
            wandb.log({"val_accuracy": val_acc}, step=0)

    model.train()
    samples_seen = global_step * gradient_accumulation_steps  # 总共处理过的样本数
    for epoch in range(resume_epoch, args.num_epochs):
        # resume 的当前 epoch 不 shuffle（保持中断前的顺序），后续 epoch 正常 shuffle
        if epoch == resume_epoch and resume_sample_idx > 0:
            skip_samples = resume_sample_idx
        else:
            random.shuffle(train_data)
            skip_samples = 0
        accum_count = 0
        pbar = tqdm(enumerate(train_data), total=len(train_data),
                    desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for i, item in pbar:
            if i < skip_samples:
                continue
            # autocast: 矩阵乘法等走 bf16 节省显存，log_softmax/logsigmoid 等自动保持 fp32 精度
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_loss = compute_per_instance_dpo_loss(
                        lm=model,
                        lm_ref=model_ref,
                        tokenizer=tokenizer,
                        beta=args.beta,
                        prompt=item["instruction"],
                        response_chosen=item["chosen_response"],
                        response_rejected=item["rejected_response"],
                        max_length=args.max_length,
                    )
                (raw_loss / gradient_accumulation_steps).backward()
            except torch.cuda.OutOfMemoryError:
                # OOM 保护：清理显存+梯度，重置累积状态，跳过该样本
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                running_loss = 0.0
                accum_count = 0
                print(f"  [OOM] Skipped sample {i}, reset accum "
                      f"(instruction len={len(item['instruction'])}, "
                      f"chosen len={len(item['chosen_response'])}, "
                      f"rejected len={len(item['rejected_response'])})")
                continue

            # NaN/Inf 检查：跳过坏样本，防止污染 optimizer state
            if not torch.isfinite(raw_loss):
                optimizer.zero_grad()
                running_loss = 0.0
                accum_count = 0
                print(f"  [NaN/Inf] Skipped sample {i}, loss={raw_loss.item()}, reset accum")
                continue
            running_loss += raw_loss.item()
            accum_count += 1
            samples_seen += 1

            # 累积够了就更新
            if accum_count == gradient_accumulation_steps:
                step_start = time.time()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                step_time = time.time() - step_start
                avg_loss = running_loss / gradient_accumulation_steps

                elapsed = time.time() - train_start_time
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0

                pbar.set_postfix(step=global_step, loss=f"{avg_loss:.4f}",
                                 grad_norm=f"{grad_norm:.2f}",
                                 eta=f"{eta/60:.0f}min")

                # ---- 日志 ----
                if global_step % args.log_every == 0:
                    print(f"[Step {global_step}/{total_steps}] "
                          f"loss={avg_loss:.4f} grad_norm={grad_norm:.2f} "
                          f"epoch={epoch+1} samples={samples_seen} "
                          f"step_time={step_time:.2f}s "
                          f"({elapsed/60:.1f}min elapsed, ~{eta/60:.0f}min remaining)")
                    log_dict = {
                        "step": global_step,
                        "loss": avg_loss,
                        "grad_norm": float(grad_norm),
                        "epoch": epoch + 1,
                        "samples_seen": samples_seen,
                        "step_time": step_time,
                    }
                    metrics_history.append(log_dict)
                    if wandb_run:
                        wandb.log(log_dict, step=global_step)

                running_loss = 0.0
                accum_count = 0

                # ---- Validation ----
                if global_step % args.eval_every == 0:
                    val_acc = compute_val_accuracy(model, model_ref, val_cached)
                    print(f"[Step {global_step}] val_accuracy={val_acc:.4f}")
                    metrics_history.append({"step": global_step, "val_accuracy": val_acc})
                    if wandb_run:
                        wandb.log({"val_accuracy": val_acc}, step=global_step)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_path = os.path.join(args.output_dir, "best_model")
                        _save_full_checkpoint(model, tokenizer, optimizer, save_path,
                                              global_step, epoch, i + 1, best_val_acc,
                                              metrics_history, wandb_run_id)
                        print(f"  -> New best model saved (acc={val_acc:.4f})")

                # ---- Save checkpoint ----
                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.output_dir, f"step_{global_step}")
                    _save_full_checkpoint(model, tokenizer, optimizer, ckpt_path,
                                          global_step, epoch, i + 1, best_val_acc,
                                          metrics_history, wandb_run_id)

        # epoch 尾部剩余梯度不丢弃
        if accum_count > 0:
            step_start = time.time()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            step_time = time.time() - step_start
            avg_loss = running_loss / accum_count
            print(f"[Step {global_step}] loss={avg_loss:.4f} grad_norm={grad_norm:.2f} "
                  f"(tail batch, {accum_count} samples)")
            log_dict = {"step": global_step, "loss": avg_loss,
                        "grad_norm": float(grad_norm), "tail_batch": True}
            metrics_history.append(log_dict)
            if wandb_run:
                wandb.log(log_dict, step=global_step)
            running_loss = 0.0

            # tail batch 后强制 eval（确保不漏最后状态），save 按 save_every 判断
            val_acc = compute_val_accuracy(model, model_ref, val_cached)
            print(f"[Step {global_step}] val_accuracy={val_acc:.4f} (tail)")
            metrics_history.append({"step": global_step, "val_accuracy": val_acc})
            if wandb_run:
                wandb.log({"val_accuracy": val_acc}, step=global_step)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(args.output_dir, "best_model")
                _save_full_checkpoint(model, tokenizer, optimizer, save_path,
                                      global_step, epoch, len(train_data), best_val_acc,
                                      metrics_history, wandb_run_id)
                print(f"  -> New best model saved (acc={val_acc:.4f})")
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"step_{global_step}")
                _save_full_checkpoint(model, tokenizer, optimizer, ckpt_path,
                                      global_step, epoch, len(train_data), best_val_acc,
                                      metrics_history, wandb_run_id)
            model.train()

    # ---- 5. 最终保存 ----
    total_time = time.time() - train_start_time

    # 先算 final eval，再保存（确保 checkpoint 里包含 final 结果）
    final_val_acc = compute_val_accuracy(model, model_ref, val_cached)
    metrics_history.append({"step": global_step, "val_accuracy": final_val_acc, "final": True})
    if wandb_run:
        wandb.log({"val_accuracy": final_val_acc, "final": True}, step=global_step)

    final_path = os.path.join(args.output_dir, "final_model")
    _save_full_checkpoint(model, tokenizer, optimizer, final_path,
                          global_step, args.num_epochs, 0, best_val_acc,
                          metrics_history, wandb_run_id)

    with open(os.path.join(args.output_dir, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/3600:.2f}h ({total_time/60:.1f}min)")
    print(f"Total steps: {global_step}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Final val accuracy: {final_val_acc:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}")

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
