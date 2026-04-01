import argparse
import json
import os
import re
import time
from pathlib import Path

from vllm import SamplingParams

from cs336_alignment.utils import init_vllm, parse_gsm8k_response
from cs336_alignment.evaluate import _normalize_cuda_visible_devices_for_vllm

PROMPTS_DIR = Path(__file__).parent / "prompts"
system_template = (PROMPTS_DIR / "zero_shot_system_prompt.prompt").read_text()

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def load_gsm8k(path):
    examples = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            match = re.search(r'####\s*(-?\d[\d,\.]*)', raw["answer"])
            gold = match.group(1).replace(",", "") if match else None
            examples.append({
                "question": raw["question"],
                "gold_answer": gold,
            })
    return examples


def format_gsm8k_prompt(example: dict, prompt_format: str = "zero_shot") -> str:
    instruction = f"{example['question']}\nAnswer:"
    if prompt_format == "alpaca":
        return ALPACA_TEMPLATE.format(instruction=instruction)
    else:
        return system_template.format(instruction=instruction)


def run_gsm8k_eval(llm, examples, prompt_format="zero_shot"):
    prompts = [format_gsm8k_prompt(ex, prompt_format) for ex in examples]

    if prompt_format == "alpaca":
        stop_seqs = ["### Instruction:", "\n\n\n"]
    else:
        stop_seqs = ["# Query:"]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=512, stop=stop_seqs,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    throughput = len(examples) / elapsed if elapsed > 0 else 0.0

    records = []
    correct = 0
    parse_failures = 0
    for ex, output in zip(examples, outputs):
        model_output = output.outputs[0].text
        predicted = parse_gsm8k_response(model_output)
        if predicted is None:
            parse_failures += 1
        is_correct = predicted == ex["gold_answer"]
        if is_correct:
            correct += 1
        records.append({
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "model_output": model_output,
            "predicted": predicted,
            "correct": is_correct,
        })

    return {
        "accuracy": correct / len(records),
        "correct": correct,
        "total": len(records),
        "parse_failures": parse_failures,
        "elapsed_sec": round(elapsed, 2),
        "throughput_ex_per_sec": round(throughput, 2),
        "prompt_format": prompt_format,
        "records": records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (overrides ASSIGNMENT5_MODEL_ID env var)")
    parser.add_argument("--prompt_format", type=str, default="zero_shot",
                        choices=["zero_shot", "alpaca"],
                        help="Prompt format: zero_shot (baseline) or alpaca (SFT)")
    parser.add_argument("--data_path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--output_path", type=str, default=None)
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
        output_path = f"outputs/gsm8k/{args.prompt_format}_results.json"

    examples = load_gsm8k(args.data_path)
    print(f"Loaded {len(examples)} GSM8K examples")
    print(f"Model: {model_path}")
    print(f"Prompt format: {args.prompt_format}")

    llm = init_vllm(model_path, device=args.device, seed=args.seed)
    results = run_gsm8k_eval(llm, examples, prompt_format=args.prompt_format)

    print(f"\nAccuracy:       {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Parse failures: {results['parse_failures']}")
    print(f"Time: {results['elapsed_sec']}s | Throughput: {results['throughput_ex_per_sec']} ex/s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")
