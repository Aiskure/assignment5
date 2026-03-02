from __future__ import annotations
"""
Expert Iteration (EI) 训练脚本（基于本仓库的 SFT 训练风格改写）。

整体流程：
1) 从 MATH 数据集中采样一批问题；
2) 用 vLLM 为每个问题生成 G 个 rollout；
3) 用 reward 函数为每个 rollout 打分，并选出每题最佳示例；
4) 用这些“伪专家示例”做一轮 SFT 更新；
5) 在验证集评估并记录 accuracy / format reward / entropy；
6) 重复上述步骤 n_ei_steps 次。
"""

import argparse
import gc
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment import utils
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ModuleNotFoundError:
    LLM = Any  # type: ignore[assignment]
    SamplingParams = Any  # type: ignore[assignment]
    HAS_VLLM = False

try:
    import wandb
    HAS_WANDB = True
except ModuleNotFoundError:
    wandb = None  # type: ignore[assignment]
    HAS_WANDB = False


PROMPT_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


@dataclass
class MathExample:
    """统一后的 MATH 样本结构，便于训练和评估阶段复用。"""
    question: str
    answer: str
    solution: str
    unique_id: str


def _resolve_model_path() -> str:
    env_model_id = os.environ.get("ASSIGNMENT5_MODEL_ID") or os.environ.get("MODEL_ID")
    if env_model_id:
        return env_model_id

    candidate_paths = [
        "/root/autodl-tmp/models/Qwen2.5-Math-1.5B",
        "/root/assignment5/models/Qwen2.5-Math-1.5B",
        "models/Qwen2.5-Math-1.5B",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return "models/Qwen2.5-Math-1.5B"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_math_dataset(path: str) -> list[MathExample]:
    """读取本地 MATH jsonl，并映射为统一字段。"""
    examples: list[MathExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            # data/math 主要使用 `problem` 字段；这里兼容可能出现的 `question` 字段。
            question = row.get("problem", row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            solution = str(row.get("solution", "")).strip()
            unique_id = str(row.get("unique_id", f"row-{i}"))
            if not question or not answer:
                continue
            examples.append(
                MathExample(
                    question=question,
                    answer=answer,
                    solution=solution,
                    unique_id=unique_id,
                )
            )
    return examples


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def canonicalize_response(raw_response: str) -> str:
    """
    将模型原始输出标准化为 r1_zero_reward_fn 期望的格式：
    - 必须包含 <think> 前缀；
    - `</think>` 与 `<answer>` 之间规范为单个空格。
    """
    text = raw_response.strip()
    if not text.startswith("<think>"):
        text = f"<think>{text}"
    text = re.sub(r"</think>\s*<answer>", "</think> <answer>", text)
    return text


def response_to_sft_output(full_response: str) -> str:
    """
    将完整 `<think> ... </answer>` 响应转成 SFT 训练需要的 output 后缀。
    注意：prompt 模板本身已经以 `<think>` 结尾，因此这里会去掉前导 `<think>`。
    """
    if full_response.startswith("<think>"):
        full_response = full_response[len("<think>"):]
    return " " + full_response.lstrip()


def create_sampling_params(
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> SamplingParams:
    """构造 vLLM 采样参数。"""
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        # 按作业要求：遇到第二段答案标签结束时停止生成。
        stop=["</answer>"],
    )
    params.include_stop_str_in_output = True
    return params


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM) -> None:
    """
    将当前可训练 policy 权重同步到已启动的 vLLM 实例中，避免频繁重启引擎。
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def select_expert_demonstrations(
    llm: LLM,
    examples: list[MathExample],
    rollouts_per_question: int,
    sampling_params: SamplingParams,
    only_positive_rollouts: bool,
) -> tuple[list[str], list[str], dict[str, float]]:
    """
    对每个问题采样 G 条 rollout，并选出 1 条“专家示例”用于后续 SFT。
    选优标准：reward > format_reward > answer_reward（按字典序比较）。
    """
    expanded_prompts: list[str] = []
    expanded_answers: list[str] = []
    for ex in examples:
        p = build_prompt(ex.question)
        expanded_prompts.extend([p] * rollouts_per_question)
        expanded_answers.extend([ex.answer] * rollouts_per_question)

    outputs = llm.generate(expanded_prompts, sampling_params)
    rollout_texts = [item.outputs[0].text for item in outputs]

    selected_prompts: list[str] = []
    selected_outputs: list[str] = []

    total = 0
    positive = 0
    formatted = 0
    accepted = 0
    accepted_positive = 0

    for i, ex in enumerate(examples):
        start = i * rollouts_per_question
        end = start + rollouts_per_question
        group_texts = rollout_texts[start:end]
        group_answers = expanded_answers[start:end]
        group_prompts = expanded_prompts[start:end]

        candidates: list[tuple[float, float, float, str, str]] = []
        for raw_text, gt_answer, prompt in zip(group_texts, group_answers, group_prompts):
            full_response = canonicalize_response(raw_text)
            reward_obj = r1_zero_reward_fn(full_response, gt_answer)
            reward = float(reward_obj["reward"])
            format_reward = float(reward_obj["format_reward"])
            answer_reward = float(reward_obj["answer_reward"])

            total += 1
            positive += int(reward > 0.0)
            formatted += int(format_reward > 0.0)
            candidates.append((reward, format_reward, answer_reward, full_response, prompt))

        # 使用多关键字打破并列，尽量让同种输入下选择过程可复现。
        best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
        best_reward, _, _, best_full_response, best_prompt = best

        if only_positive_rollouts and best_reward <= 0.0:
            continue

        accepted += 1
        accepted_positive += int(best_reward > 0.0)
        selected_prompts.append(best_prompt)
        selected_outputs.append(response_to_sft_output(best_full_response))

    metadata = {
        "rollout/total": float(total),
        "rollout/positive_rate": (positive / total) if total else 0.0,
        "rollout/format_rate": (formatted / total) if total else 0.0,
        "rollout/accepted_examples": float(accepted),
        "rollout/accepted_positive_rate": (accepted_positive / accepted) if accepted else 0.0,
    }
    return selected_prompts, selected_outputs, metadata


def sft_train_on_experts(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    prompts: list[str],
    outputs: list[str],
    micro_batch_size: int,
    local_batch_size: int,
    sft_epochs: int,
    device: torch.device,
    run: Any,
    global_step: int,
) -> tuple[dict[str, float], int]:
    """
    使用 EI 选出的伪标签数据做 SFT 更新。
    同时记录 loss 与 response token entropy，便于观察训练稳定性与模式坍塌风险。
    """
    if len(prompts) == 0:
        return {
            "train/loss": 0.0,
            "train/response_entropy": 0.0,
            "train/grad_norm": 0.0,
            "train/update_steps": 0.0,
        }, global_step

    model.train()
    optimizer.zero_grad()

    grad_accum_steps = max(local_batch_size // micro_batch_size, 1)
    n_samples = len(prompts)
    indices = list(range(n_samples))

    sum_loss = 0.0
    sum_entropy = 0.0
    n_micro_steps = 0
    n_update_steps = 0
    last_grad_norm = 0.0

    for epoch in range(sft_epochs):
        random.shuffle(indices)
        n_batches = n_samples // micro_batch_size
        pbar = tqdm(range(n_batches), desc=f"SFT-on-EI epoch {epoch + 1}/{sft_epochs}")

        for batch_i in pbar:
            start = batch_i * micro_batch_size
            end = start + micro_batch_size
            batch_ids = indices[start:end]
            batch_prompts = [prompts[idx] for idx in batch_ids]
            batch_outputs = [outputs[idx] for idx in batch_ids]

            train_batch = utils.tokenize_prompt_and_output(batch_prompts, batch_outputs, tokenizer)
            response_mask = train_batch["response_mask"].to(device)
            model_out = utils.get_response_log_probs(
                model=model,
                input_ids=train_batch["input_ids"].to(device),
                labels=train_batch["labels"].to(device),
                return_token_entropy=True,
            )
            policy_log_probs = model_out["log_probs"]
            token_entropy = model_out["token_entropy"]

            # 仅在 response token 上计算熵，用于监控“过度确定性输出”风险。
            denom = response_mask.float().sum().clamp(min=1.0)
            response_entropy = float((token_entropy * response_mask).sum().item() / denom.item())

            loss, _ = utils.sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=1.0,
            )

            if (n_micro_steps + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                n_update_steps += 1
                last_grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

            sum_loss += float(loss.item())
            sum_entropy += response_entropy
            n_micro_steps += 1
            global_step += 1

            if run is not None:
                run.log(
                    {
                        "train/loss_micro": float(loss.item()),
                        "train/response_entropy_micro": response_entropy,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "entropy": f"{response_entropy:.4f}"})

    metrics = {
        "train/loss": (sum_loss / n_micro_steps) if n_micro_steps else 0.0,
        "train/response_entropy": (sum_entropy / n_micro_steps) if n_micro_steps else 0.0,
        "train/grad_norm": last_grad_norm,
        "train/update_steps": float(n_update_steps),
    }
    return metrics, global_step


def evaluate_on_math(
    llm: LLM,
    examples: list[MathExample],
    sampling_params: SamplingParams,
    save_path: str | None = None,
) -> dict[str, float]:
    """
    在 MATH 验证集上评估模型输出，统计 accuracy / format reward / type 分布。
    """
    prompts = [build_prompt(ex.question) for ex in examples]
    outputs = llm.generate(prompts, sampling_params)
    responses = [item.outputs[0].text for item in outputs]

    records: list[dict[str, Any]] = []
    acc = 0.0
    format_reward = 0.0
    type1 = 0
    type2 = 0
    type3 = 0

    for ex, raw in zip(examples, responses):
        full_response = canonicalize_response(raw)
        result = r1_zero_reward_fn(full_response, ex.answer)

        acc += float(result["reward"])
        format_reward += float(result["format_reward"])
        if result["format_reward"] == 1.0 and result["answer_reward"] == 1.0:
            label_type = 1
            type1 += 1
        elif result["format_reward"] == 1.0 and result["answer_reward"] == 0.0:
            label_type = 2
            type2 += 1
        else:
            label_type = 3
            type3 += 1

        records.append(
            {
                "unique_id": ex.unique_id,
                "question": ex.question,
                "ground_truth": ex.answer,
                "output": full_response,
                "result": result,
                "type": label_type,
            }
        )

    n = max(len(examples), 1)
    metrics = {
        "eval/accuracy": acc / n,
        "eval/format_reward": format_reward / n,
        "eval/type1": float(type1),
        "eval/type2": float(type2),
        "eval/type3": float(type3),
    }
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在本地 MATH 数据集上运行 Expert Iteration 实验。")
    parser.add_argument("--model-path", type=str, default=_resolve_model_path())
    parser.add_argument("--train-path", type=str, default="data/math/train.jsonl")
    parser.add_argument("--test-path", type=str, default="data/math/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="cs336_alignment/ei_runs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument("--db-size", type=int, default=1024, help="Number of questions sampled per EI step.")
    parser.add_argument("--rollouts-per-question", type=int, default=8, help="G rollouts per question.")
    parser.add_argument("--sft-epochs-per-ei", type=int, default=1)

    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-max-tokens", type=int, default=1024)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-max-tokens", type=int, default=1024)

    parser.add_argument(
        "--only-positive-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, keep only questions whose best-of-G rollout has reward=1.",
    )
    parser.add_argument(
        "--save-each-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model/tokenizer after each EI step.",
    )
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-device", type=str, default="cuda:1")

    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable wandb logging if wandb is installed.",
    )
    parser.add_argument("--wandb-project", type=str, default="cs336-a5-ei")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """
    EI 主流程入口。

    代码组织按“阶段”展开，便于快速定位问题：
    1) 运行前检查与目录准备
    2) 数据加载与设备选择
    3) 模型 / vLLM / 采样器初始化
    4) step=0 基线评估
    5) EI 循环：rollout 选优 -> SFT 更新 -> 评估与落盘
    6) 训练收尾与最终指标输出
    """
    # -------------------- 阶段 1: 基础运行配置 --------------------
    # 解析命令行参数，并固定随机种子（便于复现实验）。
    args = parse_args()
    set_seed(args.seed)
    # 降低 vLLM 日志噪声，聚焦训练关键信息。
    logging.getLogger("vllm").setLevel(logging.WARNING)
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    # EI 依赖 vLLM 进行 rollout 采样；缺失时直接报错。
    if not HAS_VLLM:
        raise ModuleNotFoundError("vllm is required for expert iteration.")

    # 统一实验输出目录（评估 json、checkpoint、metrics 历史）。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 阶段 2: 数据加载 --------------------
    # 加载训练与验证集；验证集可通过 max_eval_samples 截断加速实验。
    train_examples = load_math_dataset(args.train_path)
    test_examples = load_math_dataset(args.test_path)
    if args.max_eval_samples > 0:
        test_examples = test_examples[: args.max_eval_samples]

    # 关键输入检查，避免空数据导致后续步骤失败。
    if not train_examples:
        raise ValueError(f"No training examples found in {args.train_path}")
    if not test_examples:
        raise ValueError(f"No evaluation examples found in {args.test_path}")

    # 训练模型放在 cuda:0；vLLM 通常放到另一张卡（默认 cuda:1）。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] model_path={args.model_path}")
    print(f"[INFO] train_path={args.train_path}, n_train={len(train_examples)}")
    print(f"[INFO] test_path={args.test_path}, n_eval={len(test_examples)}")
    print(f"[INFO] train_device={device}, vllm_device={args.vllm_device}")

    # -------------------- 阶段 3: 模型与采样器初始化 --------------------
    # 训练侧：HF 模型 + tokenizer + optimizer。
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 推理侧：独立 vLLM 实例，用于 rollout 和评估（速度更快）。
    llm = utils.init_vllm(
        model_id=args.model_path,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    # rollout_sampling：EI 采样阶段使用；eval_sampling：验证阶段使用。
    rollout_sampling = create_sampling_params(
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        max_tokens=args.rollout_max_tokens,
    )
    eval_sampling = create_sampling_params(
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        max_tokens=args.eval_max_tokens,
    )

    # 可选 wandb 记录；若未安装则仅本地写 json。
    run = None
    if args.use_wandb and HAS_WANDB:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    elif args.use_wandb and not HAS_WANDB:
        print("[WARN] wandb is not installed; logging to local json only.")

    metrics_history: list[dict[str, float]] = []
    global_step = 0

    # -------------------- 阶段 4: 基线评估（step=0） --------------------
    # 在还未做 EI 更新前，先得到基线准确率，便于后续比较增益。
    base_eval_path = output_dir / "eval_step_0.json"
    base_eval = evaluate_on_math(llm, test_examples, eval_sampling, save_path=str(base_eval_path))
    base_metrics = {"ei/step": 0.0, **base_eval}
    metrics_history.append(base_metrics)
    print(f"[EI step 0] eval_accuracy={base_eval['eval/accuracy']:.4f} format={base_eval['eval/format_reward']:.4f}")
    if run is not None:
        run.log(base_metrics, step=global_step)

    for ei_step in range(1, args.n_ei_steps + 1):
        # -------------------- 阶段 5.1: 采样当前 step 的训练子集 Db --------------------
        sampled_count = min(args.db_size, len(train_examples))
        sampled_examples = random.sample(train_examples, sampled_count)

        # rollout 前先同步权重，确保采样来自“当前”策略。
        load_policy_into_vllm_instance(model, llm)

        # -------------------- 阶段 5.2: rollout + reward 选优 --------------------
        # 每个问题生成 G 条候选，选择 best-of-G 作为伪专家示例。
        selected_prompts, selected_outputs, rollout_meta = select_expert_demonstrations(
            llm=llm,
            examples=sampled_examples,
            rollouts_per_question=args.rollouts_per_question,
            sampling_params=rollout_sampling,
            only_positive_rollouts=args.only_positive_rollouts,
        )

        # -------------------- 阶段 5.3: 用伪专家数据做 SFT 更新 --------------------
        train_metrics, global_step = sft_train_on_experts(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            prompts=selected_prompts,
            outputs=selected_outputs,
            micro_batch_size=args.micro_batch_size,
            local_batch_size=args.local_batch_size,
            sft_epochs=args.sft_epochs_per_ei,
            device=device,
            run=run,
            global_step=global_step,
        )

        # -------------------- 阶段 5.4: 同步后评估 --------------------
        # SFT 更新后再次同步，再在验证集上评估。
        load_policy_into_vllm_instance(model, llm)
        eval_path = output_dir / f"eval_step_{ei_step}.json"
        eval_metrics = evaluate_on_math(llm, test_examples, eval_sampling, save_path=str(eval_path))

        # 汇总本 step 指标：rollout 统计 + 训练指标 + 验证指标。
        step_metrics: dict[str, float] = {
            "ei/step": float(ei_step),
            **rollout_meta,
            **train_metrics,
            **eval_metrics,
        }
        metrics_history.append(step_metrics)

        # 每一步都覆写 metrics_history，保证中断后也能拿到阶段性结果。
        with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
            json.dump(metrics_history, f, indent=2)

        print(
            f"[EI step {ei_step}] "
            f"selected={int(rollout_meta['rollout/accepted_examples'])} "
            f"loss={train_metrics['train/loss']:.4f} "
            f"entropy={train_metrics['train/response_entropy']:.4f} "
            f"eval_acc={eval_metrics['eval/accuracy']:.4f}"
        )
        if run is not None:
            run.log(step_metrics, step=global_step)

        # 可选保存每一步 checkpoint，便于后续对比不同 EI step 模型效果。
        if args.save_each_step:
            ckpt_dir = output_dir / f"checkpoint_step_{ei_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        # 主动清理 Python 垃圾与 CUDA cache，降低长时间运行中的显存碎片风险。
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------- 阶段 6: 实验收尾 --------------------
    if run is not None:
        run.finish()

    summary = metrics_history[-1]
    print("[DONE] Expert Iteration finished.")
    print(
        f"[DONE] final_eval_accuracy={summary['eval/accuracy']:.4f}, "
        f"final_eval_format_reward={summary['eval/format_reward']:.4f}"
    )


if __name__ == "__main__":
    main()
