"""
Rejection Sampling Fine-tuning (RFT) — 数据生成脚本

对 train.jsonl 中的每道题，用 vLLM 生成 K 条 rollout，
保留 answer_reward=1（答案正确）的条目，输出为 SFT 可直接使用的 jsonl。

输出格式（与 sft.py 的 load_math_sft_pairs 兼容）：
  {"problem": <question>, "solution": <reasoning>, "answer": <answer>}

用法示例：
  uv run python -m cs336_alignment.generate_filtered_sft_data \\
      --train-path data/math/train.jsonl \\
      --output-path data/math/train_filtered.jsonl \\
      --rollouts-per-problem 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from cs336_alignment.evaluate import _normalize_cuda_visible_devices_for_vllm
_normalize_cuda_visible_devices_for_vllm()

from cs336_alignment import utils
from cs336_alignment.data_utils import extract_question_and_gt
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

try:
    from vllm import SamplingParams
    HAS_VLLM = True
except ModuleNotFoundError:
    SamplingParams = Any  # type: ignore[assignment]
    HAS_VLLM = False


PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the "
    "Assistant solves it. The Assistant first thinks about the reasoning process in "
    "the mind and then provides the User with the answer. The reasoning process is "
    "enclosed within <think> </think> and answer is enclosed within <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think> <answer> answer "
    "here </answer>.\nUser: {question}\nAssistant: <think>"
)


def _canonicalize(raw_text: str) -> str:
    text = raw_text.strip()
    if not text.startswith("<think>"):
        text = "<think>" + text
    text = re.sub(r"</think>\s*<answer>", "</think> <answer>", text)
    return text


def _parse_solution_and_answer(response: str) -> tuple[str, str] | None:
    """从规范化后的 response 中解析 solution 和 answer。

    response 格式: <think> {solution} </think> <answer> {answer} </answer>
    返回 (solution, answer)，解析失败返回 None。
    """
    m_think = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    m_answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m_think is None or m_answer is None:
        return None
    solution = m_think.group(1).strip()
    answer = m_answer.group(1).strip()
    return solution, answer


def _resolve_model_path() -> str:
    env = os.environ.get("ASSIGNMENT5_MODEL_ID") or os.environ.get("MODEL_ID")
    if env:
        return env
    for p in [
        "/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B",
        "/root/autodl-tmp/models/Qwen2.5-Math-1.5B",
        "models/Qwen2.5-Math-1.5B",
    ]:
        if os.path.exists(p):
            return p
    return "models/Qwen2.5-Math-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rejection-sampled (filtered) SFT training data."
    )
    parser.add_argument("--model-path", type=str, default=_resolve_model_path())
    parser.add_argument("--train-path", type=str, default="data/math/train.jsonl")
    parser.add_argument("--output-path", type=str, default="data/math/train_filtered.jsonl")
    parser.add_argument("--rollouts-per-problem", type=int, default=4,
                        help="每道题采样多少条 rollout（K）")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--vllm-device", type=str, default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-examples", type=int, default=0,
                        help="截断输入训练集（0=不截断），用于快速调试")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger("vllm").setLevel(logging.WARNING)
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    if not HAS_VLLM:
        raise ModuleNotFoundError("vllm is required. Install it before running this script.")

    # 加载训练集
    train_data: list[dict] = []
    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_data.append(json.loads(line))
    if args.max_train_examples > 0:
        train_data = train_data[:args.max_train_examples]
    print(f"[INFO] loaded {len(train_data)} problems from {args.train_path}")

    # 初始化 vLLM
    print(f"[INFO] loading model {args.model_path} on {args.vllm_device}")
    llm = utils.init_vllm(
        model_id=args.model_path,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
    )
    sampling_params.include_stop_str_in_output = True

    # 构造 prompt（每道题重复 K 次）
    K = args.rollouts_per_problem
    all_prompts: list[str] = []
    all_ground_truths: list[str] = []
    all_problems: list[str] = []

    n_skipped = 0
    for ex in train_data:
        try:
            question, gt = extract_question_and_gt(ex)
        except (KeyError, ValueError):
            n_skipped += 1
            continue
        prompt = PROMPT_TEMPLATE.format(question=question)
        for _ in range(K):
            all_prompts.append(prompt)
            all_ground_truths.append(gt)
            all_problems.append(question)
    if n_skipped:
        print(f"[WARN] skipped {n_skipped} invalid examples (empty question/answer)")

    print(f"[INFO] generating {len(all_prompts)} rollouts (K={K})...")
    raw_outputs = llm.generate(all_prompts, sampling_params)
    responses = [_canonicalize(o.outputs[0].text) for o in raw_outputs]

    # 过滤：只保留 answer_reward=1 的条目
    filtered: list[dict] = []
    n_correct = 0
    n_format_ok = 0

    for problem, response, gt in tqdm(
        zip(all_problems, responses, all_ground_truths),
        total=len(responses),
        desc="Scoring",
    ):
        result = r1_zero_reward_fn(response, gt)
        if result["format_reward"] == 1.0:
            n_format_ok += 1
        if result["answer_reward"] == 1.0:
            n_correct += 1
            parsed = _parse_solution_and_answer(response)
            if parsed is not None:
                solution, answer = parsed
                filtered.append({
                    "problem": problem,
                    "solution": solution,
                    "answer": answer,
                })

    total = len(responses)
    print(f"[INFO] total rollouts: {total}")
    print(f"[INFO] format OK:  {n_format_ok} ({n_format_ok/total:.1%})")
    print(f"[INFO] correct:    {n_correct} ({n_correct/total:.1%})")
    print(f"[INFO] filtered dataset size: {len(filtered)}")
    print(f"[INFO] coverage: {len(set(r['problem'] for r in filtered))}/{len(train_data)} problems have ≥1 correct rollout")

    # 保存
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in filtered:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[DONE] saved {len(filtered)} records to {output_path}")


if __name__ == "__main__":
    main()
