English | [中文](README_zh.md)

# LLM Alignment: SFT → Expert Iteration → GRPO

Implementing LLM alignment techniques on **Qwen 2.5 Math 1.5B** for mathematical reasoning. Starting from a 4.1% baseline (1024-sample val subset), reaching **40.1% on the same subset** and **46.05% on full 5000-sample validation** through iterative improvements in GRPO training.

Based on [Stanford CS336 Spring 2025 Assignment 5](https://github.com/stanford-cs336/spring2025-assignment5-alignment).

## Results

| Stage | Method | Accuracy |
|-------|--------|----------|
| Baseline | Pretrained model (no training) | 4.1% |
| SFT | Supervised fine-tuning (7500 examples) | 18.9% |
| Expert Iteration | 5-step EI (G=8, D_b=1024, E=2) | 21.9% |
| **GRPO** | **GRPO-clip + KL + masked_normalize** | **46.05%** |

SFT, Expert Iteration, and GRPO are trained independently from the pretrained base model (not sequentially). The model is too small (1.5B) for SFT→GRPO pipeline to be effective — GRPO directly on the base model works better.

See [实验记录.md](实验记录.md) for the full experimental journey, including problems encountered and how they were solved.

## Setup

### Prerequisites

- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- NVIDIA GPU(s) with CUDA support

### Install dependencies

```bash
uv sync --no-install-package flash-attn && uv sync
```

### Download model

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B
```

### Set environment variable

```bash
export ASSIGNMENT5_MODEL_ID=$(pwd)/models/Qwen2.5-Math-1.5B
```

## Usage

All training scripts are in `scripts/`. Key hyperparameters can be overridden via environment variables.

### Evaluation

Evaluate any model or checkpoint on the MATH validation set:

```bash
bash scripts/eval.sh

# Evaluate a trained checkpoint on full validation (5000 samples)
ASSIGNMENT5_MODEL_ID=outputs/grpo/run10/final_checkpoint \
MAX_EVAL_SAMPLES=0 \
bash scripts/eval.sh
```

### SFT

```bash
# Train on full dataset (7500 examples)
bash scripts/sft.sh

# Train on subset
N=512 RUN_NAME=run_sft_n512 bash scripts/sft.sh
```

### Expert Iteration

```bash
bash scripts/ei.sh

# Custom config
ROLLOUTS=8 DB_SIZE=1024 SFT_EPOCHS=2 bash scripts/ei.sh
```

### GRPO (best results)

```bash
# Default config matches our best run (run10):
# grpo_clip, lr=1.5e-5, epochs=2, kl_coef=3e-4,
# masked_normalize, use_std=True, group_filter=True
bash scripts/grpo.sh
```

Key overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `LR` | 1.5e-5 | Learning rate |
| `KL_COEF` | 3e-4 | KL penalty coefficient |
| `EPOCHS` | 2 | Off-policy epochs per rollout batch |
| `LOSS_TYPE` | grpo_clip | Loss type (`grpo_clip`, `reinforce_with_baseline`, `no_baseline`) |
| `N_STEPS` | 256 | Total GRPO training steps |
| `GROUP_FILTER` | true | Skip all-correct/all-wrong groups |
| `USE_STD` | true | Normalize advantage by group std |

## Key Findings

1. **masked_normalize >> masked_mean**: Dividing loss by `max_gen_len` instead of actual response length gave a +10.4pp improvement. Longer correct reasoning chains contribute more gradient, implicitly encouraging complete chain-of-thought.

2. **Entropy collapse is silent**: Our best run without KL penalty (run7) showed 39.8% peak accuracy during training, but the final checkpoint scored **0%** on full evaluation — it had collapsed to outputting repeated emojis. Periodic eval on a 1024-sample subset completely missed this.

3. **KL penalty is essential for off-policy GRPO**: Even a small `kl_coef=3e-4` prevents catastrophic drift. Too large (1e-3, 3e-3) constrains learning too much.

4. **Off-policy without log_ratio clamp = numerical explosion**: `exp(unbounded_log_ratio)` causes grad_norm to reach 10¹³. A simple `.clamp(-20, 20)` fixes it.

## Project Structure

```
├── cs336_alignment/       # Core implementation
│   ├── sft.py             # SFT training loop
│   ├── grpo_train_loop.py # GRPO training loop
│   ├── expert_iteration_experiment.py
│   ├── utils.py           # Loss functions, reward computation
│   ├── evaluate.py        # Evaluation with vLLM
│   └── drgrpo_grader.py   # Math answer grading
├── scripts/               # Training & evaluation scripts
│   ├── sft.sh
│   ├── ei.sh
│   ├── grpo.sh
│   └── eval.sh
├── data/math/             # MATH dataset (train 7500 / val 5000)
└── 实验记录.md             # Detailed experiment log
```
