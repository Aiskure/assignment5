"""Merge QLoRA adapter into base model (bf16) or dequantize a quantized checkpoint.

Usage:
  # SFT: base + LoRA adapter → bf16 merged
  python -m cs336_alignment.merge_qlora \
      --mode adapter \
      --base_model models/Llama-3.1-8B \
      --adapter_path outputs/sft_8b/run1_qlora_b16/adapter \
      --output_path outputs/sft_8b/run1_qlora_b16/merged_bf16

  # DPO: dequantize 4-bit → bf16
  python -m cs336_alignment.merge_qlora \
      --mode dequantize \
      --base_model outputs/dpo/run1/final_model \
      --output_path outputs/dpo/run1/merged_bf16
"""
from __future__ import annotations

import argparse
import torch
from pathlib import Path


def merge_adapter(base_model_path: str, adapter_path: str, output_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model from {base_model_path} in bf16 ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter into base model ...")
    model = model.merge_and_unload()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    print("Done (adapter merge).")


def dequantize_model(model_path: str, output_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading quantized model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )

    print("Dequantizing ...")
    model = model.dequantize()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving dequantized model to {output_path} ...")
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    print("Done (dequantize).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["adapter", "dequantize"], required=True)
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model path (adapter mode) or quantized model path (dequantize mode)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter path (only for adapter mode)")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    if args.mode == "adapter":
        if not args.adapter_path:
            raise ValueError("--adapter_path required for adapter mode")
        merge_adapter(args.base_model, args.adapter_path, args.output_path)
    else:
        dequantize_model(args.base_model, args.output_path)


if __name__ == "__main__":
    main()
