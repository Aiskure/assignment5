from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment import utils
from cs336_alignment.math_baseline import evaluate

try:
    import wandb

    HAS_WANDB = True
except ModuleNotFoundError:
    wandb = None  # type: ignore[assignment]
    HAS_WANDB = False

try:
    from vllm import LLM

    HAS_VLLM = True
except ModuleNotFoundError:
    LLM = Any  # type: ignore[assignment]
    HAS_VLLM = False


R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def _resolve_model_path() -> str:
    env_model_id = os.environ.get("ASSIGNMENT5_MODEL_ID") or os.environ.get("MODEL_ID")
    if env_model_id:
        return env_model_id

    candidate_paths = [
        "/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B",
        "/root/autodl-tmp/models/Qwen2.5-Math-1.5B",
        "/root/assignment5/models/Qwen2.5-Math-1.5B",
        "models/Qwen2.5-Math-1.5B",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return "models/Qwen2.5-Math-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training on MATH-format prompt/output pairs.")
    parser.add_argument("--model-path", type=str, default=_resolve_model_path())
    parser.add_argument("--train-path", type=str, default="data/math/train.jsonl")
    parser.add_argument("--log-directory", type=str, default="outputs/sft/run1")
    parser.add_argument("--max-train-examples", type=int, default=0,
                        help="截断训练集条数（0=不截断，用于不同规模实验）")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--log-every-local-steps", type=int, default=10)
    parser.add_argument("--eval-every-local-steps", type=int, default=1000)
    parser.add_argument(
        "--run-baseline-eval-at-step0",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--run-final-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--vllm-device", type=str, default="cuda:1")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)

    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-project", type=str, default="cs336-a5-sft")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parser.add_argument("--save-every-local-steps", type=int, default=0)
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM) -> None:
    """将训练侧 policy 权重热更新到 vLLM 实例。"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_math_sft_pairs(train_path: str, max_examples: int = 0) -> tuple[list[str], list[str]]:
    """加载 MATH 格式训练集，构造 prompt/output 对。

    MATH 数据字段：problem / solution / answer
    - prompt:  R1_ZERO_PROMPT.format(question=problem)
    - output:  {solution} </think> <answer> {answer} </answer>
    """
    prompts: list[str] = []
    outputs: list[str] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(R1_ZERO_PROMPT.format(question=row["problem"]))
            outputs.append(" " + row["solution"] + " </think> <answer> " + row["answer"] + " </answer>")
    if max_examples > 0 and len(prompts) > max_examples:
        prompts = prompts[:max_examples]
        outputs = outputs[:max_examples]
    return prompts, outputs


def should_run_eval(local_step: int, eval_every_local_steps: int) -> bool:
    if eval_every_local_steps <= 0:
        return False
    return local_step > 0 and (local_step % eval_every_local_steps == 0)


def should_save_checkpoint(local_step: int, save_every_local_steps: int) -> bool:
    if save_every_local_steps <= 0:
        return False
    return local_step > 0 and (local_step % save_every_local_steps == 0)


def _save_checkpoint(model: torch.nn.Module, tokenizer: AutoTokenizer, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def _log_wandb(run: Any, payload: dict[str, float | int], step: int) -> None:
    if run is not None:
        run.log(payload, step=step)


def main() -> None:
    args = parse_args()
    if args.micro_batch_size <= 0:
        raise ValueError("micro-batch-size must be > 0")
    if args.local_batch_size <= 0:
        raise ValueError("local-batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")

    set_seed(args.seed)
    logging.getLogger("vllm").setLevel(logging.WARNING)
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    grad_accum_steps = max(math.ceil(args.local_batch_size / args.micro_batch_size), 1)

    print(f"[INFO] MODEL_PATH={args.model_path}")
    print(f"[INFO] TRAIN_PATH={args.train_path}")
    print(
        f"[INFO] EPOCHS={args.epochs}, MICRO_BATCH_SIZE={args.micro_batch_size}, "
        f"LOCAL_BATCH_SIZE={args.local_batch_size}, GRAD_ACCUM={grad_accum_steps}, LR={args.lr}"
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)

    use_vllm_eval = (
        HAS_VLLM
        and torch.cuda.device_count() >= 2
        and (args.eval_every_local_steps > 0 or args.run_baseline_eval_at_step0 or args.run_final_eval)
    )
    llm = None
    if use_vllm_eval:
        llm = utils.init_vllm(
            model_id=args.model_path,
            device=args.vllm_device,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
    else:
        if not HAS_VLLM:
            print("[WARN] vLLM is not installed; skipping periodic eval.")
        elif torch.cuda.device_count() < 2:
            print("[WARN] <2 visible GPUs; skipping periodic eval.")

    prompts, outputs = load_math_sft_pairs(args.train_path, max_examples=args.max_train_examples)
    n_samples = len(prompts)
    log_directory = Path(args.log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.use_wandb and HAS_WANDB:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "model_path": args.model_path,
                "train_path": args.train_path,
                "epochs": args.epochs,
                "micro_batch_size": args.micro_batch_size,
                "local_batch_size": args.local_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "lr": args.lr,
                "eval_every_local_steps": args.eval_every_local_steps,
                "run_baseline_eval_at_step0": args.run_baseline_eval_at_step0,
                "run_final_eval": args.run_final_eval,
                "use_vllm_eval": use_vllm_eval,
                "save_every_local_steps": args.save_every_local_steps,
                "save_final_checkpoint": args.save_final_checkpoint,
            },
        )
    elif args.use_wandb and not HAS_WANDB:
        print("[WARN] wandb is not installed; continue without wandb logging.")

    model.train()
    local_step = 0
    opt_step = 0
    micro_since_update = 0
    last_eval_step = -1

    if use_vllm_eval and args.run_baseline_eval_at_step0:
        step_dir = log_directory / "0"
        step_dir.mkdir(parents=True, exist_ok=True)
        load_policy_into_vllm_instance(model, llm)
        accuracy, type1_num, type2_num, type3_num = evaluate(
            model_path=args.model_path,
            llm=llm,
            output_dir=str(step_dir),
            sample_seed=args.seed,
        )
        print(f"[EVAL step 0] accuracy={accuracy:.4f}")
        _log_wandb(
            wandb_run,
            {
                "eval/accuracy": float(accuracy),
                "eval/type1": int(type1_num),
                "eval/type2": int(type2_num),
                "eval/type3": int(type3_num),
            },
            step=0,
        )
        last_eval_step = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for epoch_idx in range(args.epochs):
        shuffled_indices = list(range(n_samples))
        random.shuffle(shuffled_indices)

        n_batches = math.ceil(n_samples / args.micro_batch_size)
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch_idx + 1}/{args.epochs}")

        for batch_idx in pbar:
            start = batch_idx * args.micro_batch_size
            end = min(start + args.micro_batch_size, n_samples)
            batch_indices = shuffled_indices[start:end]

            batch_prompts = [prompts[i] for i in batch_indices]
            batch_outputs = [outputs[i] for i in batch_indices]

            train_batch = utils.tokenize_prompt_and_output(batch_prompts, batch_outputs, tokenizer)
            result_batch = utils.get_response_log_probs(
                model=model,
                input_ids=train_batch["input_ids"].to(device),
                labels=train_batch["labels"].to(device),
                return_token_entropy=False,
            )
            policy_log_probs = result_batch["log_probs"]

            loss, _ = utils.sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=train_batch["response_mask"].to(device),
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=1.0,
            )

            micro_since_update += 1
            if micro_since_update == grad_accum_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                micro_since_update = 0
                _log_wandb(
                    wandb_run,
                    {
                        "train/grad_norm": float(grad_norm),
                        "train/opt_step": opt_step,
                    },
                    step=local_step + 1,
                )

            local_step += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Step": local_step})

            if local_step % max(args.log_every_local_steps, 1) == 0:
                _log_wandb(wandb_run, {"train/loss": float(loss.item())}, step=local_step)

            if use_vllm_eval and should_run_eval(local_step, args.eval_every_local_steps):
                step_dir = log_directory / str(local_step)
                step_dir.mkdir(parents=True, exist_ok=True)
                load_policy_into_vllm_instance(model, llm)
                accuracy, type1_num, type2_num, type3_num = evaluate(
                    str(step_dir),
                    llm,
                    sample_seed=args.seed,
                )
                print(f"[EVAL step {local_step}] accuracy={accuracy:.4f}")
                _log_wandb(
                    wandb_run,
                    {
                        "eval/accuracy": float(accuracy),
                        "eval/type1": int(type1_num),
                        "eval/type2": int(type2_num),
                        "eval/type3": int(type3_num),
                    },
                    step=local_step,
                )
                last_eval_step = local_step

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if should_save_checkpoint(local_step, args.save_every_local_steps):
                ckpt_dir = log_directory / "checkpoints" / f"step_{local_step}"
                _save_checkpoint(model, tokenizer, ckpt_dir)

    # 尾步保护：补最后不足一个 grad_accum_steps 的梯度。
    if micro_since_update > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1
        _log_wandb(
            wandb_run,
            {
                "train/grad_norm": float(grad_norm),
                "train/opt_step": opt_step,
            },
            step=local_step,
        )
        print(f"[INFO] applied tail optimizer step at local_step={local_step}")

    if use_vllm_eval and args.run_final_eval and local_step != last_eval_step:
        step_dir = log_directory / str(local_step)
        step_dir.mkdir(parents=True, exist_ok=True)
        load_policy_into_vllm_instance(model, llm)
        accuracy, type1_num, type2_num, type3_num = evaluate(
            model_path=args.model_path,
            llm=llm,
            output_dir=str(step_dir),
            sample_seed=args.seed,
        )
        print(f"[EVAL final step {local_step}] accuracy={accuracy:.4f}")
        _log_wandb(
            wandb_run,
            {
                "eval/accuracy": float(accuracy),
                "eval/type1": int(type1_num),
                "eval/type2": int(type2_num),
                "eval/type3": int(type3_num),
            },
            step=local_step,
        )

    if args.save_final_checkpoint:
        final_ckpt_dir = log_directory / "final_checkpoint"
        _save_checkpoint(model, tokenizer, final_ckpt_dir)
        print(f"[INFO] saved final checkpoint to {final_ckpt_dir}")

    if wandb_run is not None:
        wandb_run.finish()

    print("[DONE] SFT training finished.")


if __name__ == "__main__":
    main()
