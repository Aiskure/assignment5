#!/usr/bin/env python3
"""Download HH-RLHF dataset from HuggingFace."""
import os
from pathlib import Path
from datasets import load_dataset

def download_hh_rlhf(output_dir: str = "data/hh"):
    """Download all HH-RLHF splits."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset splits to download
    splits = [
        "harmless-base",
        "helpful-base", 
        "helpful-online",
        "helpful-rejection-sampled",
    ]
    
    for split in splits:
        print(f"\nDownloading {split}...")
        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "Anthropic/hh-rlhf",
                split=split,
                trust_remote_code=True,
            )
            
            # Save as JSONL
            output_file = output_path / f"{split}.jsonl"
            print(f"Saving to {output_file}...")
            
            with open(output_file, "w") as f:
                for example in dataset:
                    import json
                    f.write(json.dumps(example) + "\n")
            
            print(f"Saved {len(dataset)} examples to {output_file}")
            
        except Exception as e:
            print(f"Error downloading {split}: {e}")
            continue
    
    print(f"\nAll downloads complete! Data saved to {output_path.absolute()}")

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/hh"
    download_hh_rlhf(output_dir)
