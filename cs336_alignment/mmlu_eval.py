import argparse
import json
import os
import time
from pathlib import Path

from vllm import SamplingParams

from cs336_alignment.utils import init_vllm, parse_mmlu_response
from cs336_alignment.evaluate import _normalize_cuda_visible_devices_for_vllm

PROMPTS_DIR = Path("cs336_alignment/prompts")
mmlu_template = (PROMPTS_DIR / "mmlu.prompt").read_text()
system_template = (PROMPTS_DIR / "zero_shot_system_prompt.prompt").read_text()

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

LETTERS = ["A", "B", "C", "D"]


def load_mmlu(path):
    examples = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            examples.append({
                "subject": raw["subject"],
                "question": raw["question"],
                "options": raw["choices"],
                "answer": LETTERS[raw["answer"]],
            })
    return examples


def format_mmlu_instruction(example: dict) -> str:
    return mmlu_template.format(
        subject=example["subject"],
        question=example["question"],
        options=example["options"],
    )


def format_mmlu_prompt(example: dict, prompt_format: str = "zero_shot") -> str:
    instruction = format_mmlu_instruction(example)
    if prompt_format == "alpaca":
        return ALPACA_TEMPLATE.format(instruction=instruction)
    else:
        return system_template.format(instruction=instruction)


def run_mmlu_eval(llm, examples, prompt_format="zero_shot"):
    prompts = [format_mmlu_prompt(ex, prompt_format) for ex in examples]

    if prompt_format == "alpaca":
        stop_seqs = ["### Instruction:", "\n\n\n"]
    else:
        stop_seqs = ["# Query:"]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=128, stop=stop_seqs,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    records = []
    correct = 0
    for ex, output in zip(examples, outputs):
        model_output = output.outputs[0].text
        predicted = parse_mmlu_response(ex, model_output)
        is_correct = predicted == ex["answer"]
        if is_correct:
            correct += 1
        records.append({
            "subject": ex["subject"],
            "question": ex["question"],
            "options": ex["options"],
            "gold_answer": ex["answer"],
            "model_output": model_output,
            "predicted": predicted,
            "correct": is_correct,
        })

    throughput = len(records) / elapsed if elapsed > 0 else 0.0
    return {
        "accuracy": correct / len(records),
        "correct": correct,
        "total": len(records),
        "elapsed_sec": round(elapsed, 2),
        "throughput_ex_per_sec": round(throughput, 2),
        "prompt_format": prompt_format,
        "records": records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MMLU evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (overrides ASSIGNMENT5_MODEL_ID env var)")
    parser.add_argument("--prompt_format", type=str, default="zero_shot",
                        choices=["zero_shot", "alpaca"],
                        help="Prompt format: zero_shot (baseline) or alpaca (SFT)")
    parser.add_argument("--data_path", type=str, default="data/MMLU/mmlu_test.jsonl")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    _normalize_cuda_visible_devices_for_vllm()
    args = parse_args()

    model_path = args.model_path or os.environ.get(
        "ASSIGNMENT5_MODEL_ID",
        "/scratch/users/nus/e1553316/assignment5/models/Llama-3.1-8B",
    )

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = f"outputs/mmlu/{args.prompt_format}_results.json"

    examples = load_mmlu(args.data_path)
    print(f"Loaded {len(examples)} MMLU examples")
    print(f"Model: {model_path}")
    print(f"Prompt format: {args.prompt_format}")

    llm = init_vllm(model_path, device=args.device, seed=args.seed)
    results = run_mmlu_eval(llm, examples, prompt_format=args.prompt_format)

    print(f"\nAccuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Time: {results['elapsed_sec']}s | Throughput: {results['throughput_ex_per_sec']} ex/s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")
