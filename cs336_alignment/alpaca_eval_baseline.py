import argparse
import json
import os
import time
from pathlib import Path
from vllm import LLM, SamplingParams

from cs336_alignment.evaluate import _normalize_cuda_visible_devices_for_vllm

PROMPTS_DIR = Path(__file__).parent / "prompts"
system_template = (PROMPTS_DIR / "zero_shot_system_prompt.prompt").read_text()

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def load_alpaca_eval(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_alpaca_prompt(example: dict, prompt_format: str = "zero_shot") -> str:
    if prompt_format == "alpaca":
        return ALPACA_TEMPLATE.format(instruction=example["instruction"])
    else:
        return system_template.format(instruction=example["instruction"])


def generate_outputs(llm, examples, prompt_format="zero_shot", generator_name="Llama-3.1-8B"):
    prompts = [format_alpaca_prompt(ex, prompt_format) for ex in examples]

    if prompt_format == "alpaca":
        stop_seqs = ["### Instruction:", "\n\n\n"]
    else:
        stop_seqs = ["# Query:"]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=2048, stop=stop_seqs,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    throughput = len(examples) / elapsed if elapsed > 0 else 0.0
    print(f"Throughput: {throughput:.2f} examples/second ({elapsed:.1f}s total)")

    results = []
    for ex, output in zip(examples, outputs):
        results.append({
            "instruction": ex["instruction"],
            "output": output.outputs[0].text,
            "generator": generator_name,
            "dataset": ex["dataset"],
        })
    return results, {"elapsed_sec": round(elapsed, 2), "throughput_ex_per_sec": round(throughput, 2)}


def parse_args():
    parser = argparse.ArgumentParser(description="AlpacaEval output generation")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prompt_format", type=str, default="zero_shot",
                        choices=["zero_shot", "alpaca"])
    parser.add_argument("--generator_name", type=str, default=None,
                        help="Generator name in output JSON (auto-set if omitted)")
    parser.add_argument("--data_path", type=str, default="data/AlpacaEval/alpaca_eval.jsonl")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    _normalize_cuda_visible_devices_for_vllm()
    args = parse_args()

    model_path = args.model_path or os.environ.get(
        "ASSIGNMENT5_MODEL_ID",
        "/scratch/users/nus/e1553316/assignment5/models/Llama-3.1-8B",
    )

    generator_name = args.generator_name or (
        "Llama-3.1-8B-SFT" if args.prompt_format == "alpaca" else "Llama-3.1-8B"
    )

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = f"outputs/alpaca_eval/{args.prompt_format}_model_outputs.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Prompt format: {args.prompt_format}")
    print(f"Generator: {generator_name}")

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096)
    examples = load_alpaca_eval(args.data_path)
    results, timing = generate_outputs(llm, examples, args.prompt_format, generator_name)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} outputs to {output_path}")
    print(f"Timing: {timing}")
