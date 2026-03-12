from __future__ import annotations
"""
Expert Iteration (EI) 训练脚本（基于本仓库的 SFT 训练风格改写）。

整体流程：
1) 从 MATH 数据集中采样一批问题；
2) 用 vLLM 为每个问题生成 G 个 rollout；
3) 用 reward 函数为每个 rollout 打分，并选出每题最佳示例；
4) 用这些"伪专家示例"做一轮 SFT 更新；
5) 在验证集评估并记录 accuracy / format reward / entropy；
6) 重复上述步骤 n_ei_steps 次。
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# NSCC PBS 节点上 CUDA_VISIBLE_DEVICES 可能是 UUID 格式（"GPU-..."），
# 需要在 vLLM 初始化前转换为数字索引，否则 vLLM 无法识别设备。
from cs336_alignment.evaluate import _normalize_cuda_visible_devices_for_vllm
_normalize_cuda_visible_devices_for_vllm()

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
        "/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B",
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
    """构造 vLLM 采样参数。

    注意：
    - `min_tokens=4`：避免模型生成空字符串 rollout，空串会导致 reward 解析异常或 NaN 损失。
      老师在作业说明中特别提醒了这一点。
    - `include_stop_str_in_output=True`：让 stop string `</answer>` 出现在输出文本中，
      便于 `r1_zero_reward_fn` 正确解析 answer 标签。
    """
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=4,          # 防止生成空 rollout 导致后续 reward 解析出错
        stop=["</answer>"],    # 遇到答案结束标签时停止
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
) -> tuple[list[str], list[str], list[dict], dict[str, float]]:
    """
    对每个问题采样 G 条 rollout，保留所有 answer_reward=1 且 format_reward=1 的正确 rollout 作为伪专家示例。

    这里实现的是老师 Algorithm 2 中的过滤逻辑：
    - 对每道题生成 G 条候选输出
    - 用 reward 函数打分
    - **保留所有** answer_reward=1 且 format_reward=1 的 rollout（而不是每题只取 best-of-G 一条）
    - 错误 rollout 全部丢弃

    返回：
    - selected_prompts:  过滤后的 prompt 列表（一道题可能贡献多条）
    - selected_outputs:  对应的 SFT 输出后缀列表
    - dsft_records:      每条被选中 rollout 的原始记录（用于落盘分析）
    - metadata:          本轮 rollout 的统计信息（positive rate、coverage 等）
    """
    # ---- 展平：每道题重复 G 次，批量送入 vLLM ----
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
    dsft_records: list[dict] = []   # 用于落盘：每条正确 rollout 的详细信息

    total = 0
    positive = 0       # answer_reward=1 且 format_reward=1 的 rollout 数
    formatted = 0      # format_reward=1 的 rollout 数
    covered_problems = 0   # 至少有 1 条正确 rollout 的题目数

    for i, ex in enumerate(examples):
        start = i * rollouts_per_question
        end = start + rollouts_per_question
        group_texts = rollout_texts[start:end]
        group_prompt = expanded_prompts[start]        # 同一题 prompt 相同，取第一条即可
        group_gt = expanded_answers[start]

        problem_has_correct = False   # 记录当前题目是否有至少 1 条正确 rollout

        for raw_text in group_texts:
            full_response = canonicalize_response(raw_text)
            reward_obj = r1_zero_reward_fn(full_response, group_gt)
            answer_reward = float(reward_obj["answer_reward"])
            format_reward = float(reward_obj["format_reward"])
            reward = float(reward_obj["reward"])

            total += 1
            if format_reward == 1.0:
                formatted += 1
            if answer_reward == 1.0 and format_reward == 1.0:
                # 只保留答案正确且格式正确的 rollout
                positive += 1
                problem_has_correct = True
                sft_out = response_to_sft_output(full_response)
                selected_prompts.append(group_prompt)
                selected_outputs.append(sft_out)
                # 记录详情供落盘分析
                dsft_records.append({
                    "unique_id": ex.unique_id,
                    "prompt": group_prompt,
                    "response": full_response,
                    "reward": reward,
                    "format_reward": format_reward,
                    "answer_reward": answer_reward,
                })

        if problem_has_correct:
            covered_problems += 1

    metadata = {
        "rollout/total": float(total),
        # 答案+格式双正确率（answer_reward=1 且 format_reward=1 占全部 rollout 的比例）
        "rollout/positive_rate": (positive / total) if total else 0.0,
        # 格式合规率（format_reward=1 占全部 rollout 的比例）
        "rollout/format_rate": (formatted / total) if total else 0.0,
        # 本轮 D_sft 的样本数（所有双正确 rollout 的总数）
        "rollout/accepted_examples": float(len(selected_prompts)),
        # 有至少 1 条正确 rollout 的题目覆盖数
        "rollout/covered_problems": float(covered_problems),
        # 覆盖率 = 有正确解的题目数 / 本轮采样题目总数
        "rollout/coverage_rate": (covered_problems / len(examples)) if examples else 0.0,
    }
    return selected_prompts, selected_outputs, dsft_records, metadata


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

    # 用 math.ceil 向上取整，确保 local_batch_size 不能整除时也不会少做 accumulation step。
    # 例如 local_batch_size=32, micro_batch_size=3 → ceil(32/3)=11，而非 floor=10。
    grad_accum_steps = max(math.ceil(local_batch_size / micro_batch_size), 1)
    n_samples = len(prompts)
    indices = list(range(n_samples))

    sum_loss = 0.0
    sum_entropy = 0.0
    n_micro_steps = 0
    n_update_steps = 0
    last_grad_norm = 0.0

    for epoch in range(sft_epochs):
        random.shuffle(indices)
        # 用 math.ceil 向上取整：保证最后一个不满 micro_batch_size 的尾部样本也不会被丢弃。
        # EI 每步筛出的样本数不固定，尾部丢弃问题会更明显。
        n_batches = math.ceil(n_samples / micro_batch_size)
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

            # 仅在 response token 上计算熵，用于监控"过度确定性输出"风险。
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

    # 尾步保护：最后不足 grad_accum_steps 的 microbatches 也执行一次参数更新。
    if n_micro_steps % grad_accum_steps != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        n_update_steps += 1
        last_grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

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
    parser.add_argument(
        "--validation-path",
        "--test-path",
        dest="validation_path",
        type=str,
        default="data/math/validation.jsonl",
    )
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

    # 注：已移除 --only-positive-rollouts 参数。
    # 新版 select_expert_demonstrations() 固定只保留 answer_reward=1 且 format_reward=1 的 rollout，
    # 这与老师 Algorithm 2 的语义严格对齐，不再提供关闭开关。
    parser.add_argument(
        "--save-each-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model/tokenizer after each EI step.",
    )
    # 与 baseline / SFT 实验统一使用 1024 条验证样本，便于横向比较 accuracy 曲线。
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-device", type=str, default="cuda:1")

    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable wandb logging if wandb is installed.",
    )
    parser.add_argument("--wandb-project", type=str, default="cs336-a5")
    parser.add_argument("--wandb-group", type=str, default="ei")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """
    EI 主流程入口。

    代码组织按"阶段"展开，便于快速定位问题：
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
    test_examples = load_math_dataset(args.validation_path)
    if args.max_eval_samples > 0:
        test_examples = test_examples[: args.max_eval_samples]

    # 关键输入检查，避免空数据导致后续步骤失败。
    if not train_examples:
        raise ValueError(f"No training examples found in {args.train_path}")
    if not test_examples:
        raise ValueError(f"No evaluation examples found in {args.validation_path}")

    # 训练模型放在 cuda:0；vLLM 通常放到另一张卡（默认 cuda:1）。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] model_path={args.model_path}")
    print(f"[INFO] train_path={args.train_path}, n_train={len(train_examples)}")
    print(f"[INFO] validation_path={args.validation_path}, n_eval={len(test_examples)}")
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
            group=args.wandb_group,
            name=args.wandb_run_name,
            config=vars(args),
            dir=str(output_dir),
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

        # rollout 前先同步权重，确保采样来自"当前"策略。
        load_policy_into_vllm_instance(model, llm)

        # -------------------- 阶段 5.2: rollout + reward 过滤 --------------------
        # 每个问题生成 G 条候选，保留所有 answer_reward=1 且 format_reward=1 的正确 rollout 作为 D_sft。
        # （旧版为每题 best-of-G 取 1 条；新版符合老师 Algorithm 2 的过滤语义。）
        selected_prompts, selected_outputs, dsft_records, rollout_meta = select_expert_demonstrations(
            llm=llm,
            examples=sampled_examples,
            rollouts_per_question=args.rollouts_per_question,
            sampling_params=rollout_sampling,
        )

        # 将本步 D_sft 落盘，便于后续分析：哪些题目被筛出、哪些一条也没有。
        # 文件格式：jsonl，每行一条被选中的正确 rollout 记录。
        dsft_path = output_dir / f"ei_sft_step_{ei_step}.jsonl"
        with open(dsft_path, "w", encoding="utf-8") as f:
            for rec in dsft_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[EI step {ei_step}] D_sft saved to {dsft_path} ({len(dsft_records)} records)")

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
            f"D_sft={int(rollout_meta['rollout/accepted_examples'])} "
            f"coverage={rollout_meta['rollout/coverage_rate']:.1%} "
            f"pos_rate={rollout_meta['rollout/positive_rate']:.1%} "
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
