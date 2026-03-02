from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
from typing import Any

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment import utils
from cs336_alignment.math_baseline import evaluate

try:
    from vllm import LLM

    HAS_VLLM = True
except ModuleNotFoundError:
    LLM = Any  # type: ignore[assignment]
    HAS_VLLM = False


def _resolve_model_path() -> str:
    env_model_id = os.environ.get("ASSIGNMENT5_MODEL_ID") or os.environ.get("MODEL_ID")
    if env_model_id:
        return env_model_id

    candidate_paths = [
        "/root/autodl-tmp/models/Qwen2.5-Math-1.5B",
        "models/Qwen2.5-Math-1.5B",
        "/root/assignment5/models/Qwen2.5-Math-1.5B",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return "models/Qwen2.5-Math-1.5B"


# ========================= 可直接修改的实验参数（无 argparse） =========================
MODEL_PATH = _resolve_model_path()
TRAIN_PATH = "data/gsm8k/train.jsonl"
LOG_DIRECTORY = "cs336_alignment/sft_eval"

SEED = 42
EPOCHS = 3
MICRO_BATCH_SIZE = 1
LOCAL_BATCH_SIZE = 32
LR = 1e-5

# 训练日志与评估频率
LOG_EVERY_LOCAL_STEPS = 10
EVAL_EVERY_LOCAL_STEPS = 1000
RUN_BASELINE_EVAL_AT_STEP0 = False

# vLLM 相关设置
VLLM_DEVICE = "cuda:1"
VLLM_GPU_MEMORY_UTILIZATION = 0.8


R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


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


def load_gsm8k_sft_pairs(train_path: str) -> tuple[list[str], list[str]]:
    """加载 GSM8K 训练集，并构造 prompt/output 对。"""
    prompts: list[str] = []
    outputs: list[str] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(R1_ZERO_PROMPT.format(question=row["question"]))
            outputs.append(" " + row["answer"].replace("#### ", " </think> <answer> ") + " </answer>")
    return prompts, outputs


def should_run_eval(local_step: int) -> bool:
    if EVAL_EVERY_LOCAL_STEPS <= 0:
        return False
    if RUN_BASELINE_EVAL_AT_STEP0 and local_step == 0:
        return True
    return local_step > 0 and (local_step % EVAL_EVERY_LOCAL_STEPS == 0)


def main() -> None:
    set_seed(SEED)
    logging.getLogger("vllm").setLevel(logging.WARNING)
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] MODEL_PATH={MODEL_PATH}")
    print(f"[INFO] TRAIN_PATH={TRAIN_PATH}")
    print(
        f"[INFO] EPOCHS={EPOCHS}, MICRO_BATCH_SIZE={MICRO_BATCH_SIZE}, "
        f"LOCAL_BATCH_SIZE={LOCAL_BATCH_SIZE}, LR={LR}"
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    optimizer.zero_grad(set_to_none=True)

    use_vllm_eval = HAS_VLLM and torch.cuda.device_count() >= 2 and EVAL_EVERY_LOCAL_STEPS > 0
    llm = None
    if use_vllm_eval:
        llm = utils.init_vllm(
            model_id=MODEL_PATH,
            device=VLLM_DEVICE,
            seed=SEED,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )
    else:
        if not HAS_VLLM:
            print("[WARN] vLLM is not installed; skipping periodic eval.")
        elif torch.cuda.device_count() < 2:
            print("[WARN] <2 visible GPUs; skipping periodic eval.")

    prompts, outputs = load_gsm8k_sft_pairs(TRAIN_PATH)
    n_samples = len(prompts)
    grad_accum_steps = max(LOCAL_BATCH_SIZE // MICRO_BATCH_SIZE, 1)
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    wandb_run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "cs336-a5-sft"),
        entity=os.getenv("WANDB_ENTITY"),
        name=os.getenv("WANDB_RUN_NAME"),
        config={
            "model_path": MODEL_PATH,
            "train_path": TRAIN_PATH,
            "epochs": EPOCHS,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "gradient_accumulation_steps": grad_accum_steps,
            "lr": LR,
            "eval_every_local_steps": EVAL_EVERY_LOCAL_STEPS,
            "run_baseline_eval_at_step0": RUN_BASELINE_EVAL_AT_STEP0,
            "use_vllm_eval": use_vllm_eval,
        },
    )

    model.train()
    local_step = 0
    opt_step = 0

    for epoch_idx in range(EPOCHS):
        # 优化点 1：每个 epoch 打乱顺序，减少顺序偏置。
        shuffled_indices = list(range(n_samples))
        random.shuffle(shuffled_indices)

        # 优化点 2：使用 ceil，不丢弃最后一个不满 micro-batch 的尾部样本。
        n_batches = math.ceil(n_samples / MICRO_BATCH_SIZE)
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch_idx + 1}/{EPOCHS}")

        for batch_idx in pbar:
            start = batch_idx * MICRO_BATCH_SIZE
            end = min(start + MICRO_BATCH_SIZE, n_samples)
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

            if (local_step + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # 优化点 3：set_to_none=True 可减少显存写入和开销。
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                wandb.log(
                    {
                        "train/grad_norm": float(grad_norm),
                        "train/opt_step": opt_step,
                    },
                    step=local_step,
                )

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Step": local_step})
            if local_step % LOG_EVERY_LOCAL_STEPS == 0:
                wandb.log({"train/loss": float(loss.item())}, step=local_step)

            # 优化点 4：默认不在 step=0 评估，避免启动即跑一次完整评估耗时。
            if use_vllm_eval and should_run_eval(local_step):
                save_directory = f"{LOG_DIRECTORY}/{local_step}"
                os.makedirs(save_directory, exist_ok=True)

                load_policy_into_vllm_instance(model, llm)
                accuracy, type1_num, type2_num, type3_num = evaluate(save_directory, llm)

                wandb.log(
                    {
                        "eval/accuracy": float(accuracy),
                        "eval/type1": int(type1_num),
                        "eval/type2": int(type2_num),
                        "eval/type3": int(type3_num),
                    },
                    step=local_step,
                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            local_step += 1

    wandb_run.finish()
    print("[DONE] SFT training finished.")


if __name__ == "__main__":
    main()
