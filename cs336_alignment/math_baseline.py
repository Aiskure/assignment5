from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
from typing import Any, Callable

import torch

try:
    from drgrpo_grader import r1_zero_reward_fn
    from data_utils import extract_question_and_gt
except Exception:
    from .drgrpo_grader import r1_zero_reward_fn
    from .data_utils import extract_question_and_gt


def _normalize_cuda_visible_devices_for_vllm() -> None:
    """Convert UUID-style CUDA_VISIBLE_DEVICES to numeric indices for vLLM."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices or "GPU-" not in visible_devices:
        return

    raw_tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if not raw_tokens:
        return

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to map CUDA_VISIBLE_DEVICES UUIDs to indices (%s).", exc
        )
        return

    uuid_to_index: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            uuid_to_index[parts[1]] = parts[0]

    mapped_tokens: list[str] = []
    for token in raw_tokens:
        if token.isdigit():
            mapped_tokens.append(token)
            continue
        mapped = uuid_to_index.get(token)
        if mapped is None:
            logging.getLogger(__name__).warning(
                "Cannot map CUDA device token '%s'; keep original CUDA_VISIBLE_DEVICES.",
                token,
            )
            return
        mapped_tokens.append(mapped)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(mapped_tokens)


_normalize_cuda_visible_devices_for_vllm()
try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    LLM = Any  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]


logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

DEFAULT_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def _require_vllm() -> None:
    if SamplingParams is None:
        raise ModuleNotFoundError(
            "vllm is required for math_baseline evaluation. Install vllm before calling evaluate()."
        )


def _resolve_model_path() -> str | None:
    model_path = os.environ.get("ASSIGNMENT5_MODEL_ID") or os.environ.get("MODEL_ID")
    if model_path:
        return model_path

    candidate_paths = [
        "/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B",
        "/root/autodl-tmp/models/Qwen2.5-Math-1.5B",
        "/root/assignment5/models/Qwen2.5-Math-1.5B",
        "models/Qwen2.5-Math-1.5B",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def generate_vllm_outputs(
    vllm_model: LLM,
    prompts: list[str],
    eval_sampling_params: SamplingParams,
) -> list[str]:
    _require_vllm()
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    return [output.outputs[0].text for output in outputs]


def _load_eval_examples(
    dataset_path: str,
    max_eval_samples: int,
    sampling_strategy: str,
    sample_seed: int,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            try:
                question, answer = extract_question_and_gt(row)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Skip invalid row at %s:%d (%s)", dataset_path, line_idx, exc
                )
                continue

            examples.append(
                {
                    "unique_id": str(row.get("unique_id", f"row-{line_idx}")),
                    "question": question,
                    "ground_truth": answer,
                }
            )

    if not examples:
        raise ValueError(f"No valid evaluation examples found in {dataset_path}")

    if max_eval_samples > 0 and len(examples) > max_eval_samples:
        if sampling_strategy == "first_n":
            examples = examples[:max_eval_samples]
        elif sampling_strategy == "seeded_random":
            rng = random.Random(sample_seed)
            indices = rng.sample(range(len(examples)), max_eval_samples)
            indices.sort()
            examples = [examples[idx] for idx in indices]
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")
    return examples


def evaluate(
    model_path: str,
    llm: LLM | None = None,
    rl: bool = False,
    reward_fn: Callable[[str, str], dict[str, float]] | None = None,
    prompt: str | None = None,
    dataset_path: str = "data/math/validation.jsonl",
    max_eval_samples: int = 1024,
    output_dir: str | None = None,
    sampling_strategy: str = "seeded_random",
    sample_seed: int = 42,
) -> tuple[float, float] | tuple[float, int, int, int]:
    _require_vllm()
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
    )
    sampling_params.include_stop_str_in_output = True

    if llm is None:
        llm = LLM(model=model_path, gpu_memory_utilization=0.8, dtype=torch.float16)
    if reward_fn is None:
        reward_fn = r1_zero_reward_fn
    if prompt is None:
        prompt = DEFAULT_PROMPT

    examples = _load_eval_examples(
        dataset_path=dataset_path,
        max_eval_samples=max_eval_samples,
        sampling_strategy=sampling_strategy,
        sample_seed=sample_seed,
    )
    prompts = [prompt.format(question=ex["question"]) for ex in examples]
    answers = [ex["ground_truth"] for ex in examples]
    outputs = generate_vllm_outputs(llm, prompts, sampling_params)

    acc = 0.0
    format_reward = 0.0
    type1_num = 0
    type2_num = 0
    type3_num = 0
    records: list[dict[str, Any]] = []

    for ex, gt, raw_output in zip(examples, answers, outputs):
        full_response = raw_output
        if not full_response.lstrip().startswith("<think>"):
            full_response = "<think>" + full_response
        full_response = re.sub(r"</think>\s*<answer>", "</think> <answer>", full_response)

        result = reward_fn(full_response, gt)
        if result["format_reward"] == 1.0 and result["answer_reward"] == 1.0:
            sample_type = 1
            type1_num += 1
        elif result["format_reward"] == 1.0 and result["answer_reward"] == 0.0:
            sample_type = 2
            type2_num += 1
        else:
            sample_type = 3
            type3_num += 1

        records.append(
            {
                "unique_id": ex["unique_id"],
                "question": ex["question"],
                "ground_truth": gt,
                "raw_output": raw_output,
                "output": full_response,
                "result": result,
                "type": sample_type,
            }
        )
        acc += float(result["reward"])
        format_reward += float(result["format_reward"])

    accuracy = acc / len(outputs)
    avg_format_reward = format_reward / len(outputs)

    save_dir = output_dir or model_path
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test_log.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if rl:
        return accuracy, avg_format_reward
    return accuracy, type1_num, type2_num, type3_num


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline evaluation on MATH validation set.")
    parser.add_argument("--model-path", type=str, default=_resolve_model_path())
    parser.add_argument("--dataset-path", type=str, default="data/math/validation.jsonl")
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="seeded_random",
        choices=["seeded_random", "first_n"],
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.model_path:
        raise FileNotFoundError(
            "Model path not found. Set ASSIGNMENT5_MODEL_ID or MODEL_ID to your local model directory."
        )

    accuracy, type1_num, type2_num, type3_num = evaluate(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        max_eval_samples=args.max_eval_samples,
        output_dir=args.output_dir,
        sampling_strategy=args.sampling_strategy,
        sample_seed=args.sample_seed,
    )
    print(
        f"[DONE] accuracy={accuracy:.6f}, "
        f"type1/type2/type3={type1_num}/{type2_num}/{type3_num}, "
        f"dataset={args.dataset_path}, max_eval_samples={args.max_eval_samples}, "
        f"sampling_strategy={args.sampling_strategy}, sample_seed={args.sample_seed}"
    )
