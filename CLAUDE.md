# CLAUDE.md - Project Context for Claude Code

## 🎓 Learning Context

This is **Stanford CS336 Spring 2025 Assignment 5: Alignment** - a publicly available course assignment. The user is completing this assignment **for self-learning and skill development purposes**, not for academic credit or grade submission.

**Claude's Role**: 协助者（tutor），而非直接工作者。最终目的是提升用户能力。
- **不要直接写完整实现**，而是引导用户自己完成
- 提供思路、概念解释、伪代码、关键提示
- 当用户写好代码后，帮助 review 和调试
- 用户卡住时，给出最小提示（hint），而非答案
- 可以指出需要用到的 API / 函数 / 模式，但让用户自己组装
- 当用户明确要求"帮我写"时，才直接写代码

## Project Overview

This project implements LLM alignment techniques (SFT, Expert Iteration, GRPO) using Qwen 2.5 Math 1.5B.

**Important**: We use **GSM8K** instead of MATH as the training/evaluation dataset (MATH is not open-source).

## Repository Structure

```
assignment5/
├── cs336_alignment/           # Main package
│   ├── __init__.py
│   ├── drgrpo_grader.py       # Math answer grading (r1_zero_reward_fn, question_only_reward_fn)
│   └── prompts/               # Prompt templates
│       ├── r1_zero.prompt     # <think>/<answer> format (primary)
│       ├── question_only.prompt
│       ├── alpaca_sft.prompt
│       └── zero_shot_system_prompt.prompt
├── tests/                     # Test suite
│   ├── adapters.py            # *** MAIN FILE TO IMPLEMENT *** (all adapter functions)
│   ├── conftest.py            # Pytest fixtures and snapshot testing utilities
│   ├── common.py              # Shared constants (FIXTURES_PATH)
│   ├── test_sft.py            # SFT-related tests
│   ├── test_grpo.py           # GRPO-related tests
│   ├── test_metrics.py        # MMLU/GSM8K parsing tests (optional)
│   ├── test_data.py           # Dataset/dataloader tests (optional)
│   ├── test_dpo.py            # DPO tests (optional)
│   └── _snapshots/            # Expected test outputs (.npz files)
├── scripts/                   # Evaluation scripts
├── plan.md                    # Detailed study plan (Chinese)
├── 实验记录.md                 # Experiment log
├── 学习记录.md                 # Learning notes
├── pyproject.toml             # Project config (uv, pytest)
└── test_and_make_submission.sh # Test + zip for submission
```

## Key Information

- **Model**: Qwen 2.5 Math 1.5B at `/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B`
- **Dataset**: **GSM8K** (替代 MATH，因 MATH 不开源)
- **Package manager**: `uv` (not pip)
- **Python**: 3.11 or 3.12 (not 3.13)

## Training Environment (NSCC)

- **集群**: NSCC (National Supercomputing Centre Singapore)
- **GPU**: 4x NVIDIA A100 40GB per job
- **总显存**: 160GB per job
- **注意事项**:
  - A100 40GB 相比 H100 显存较小，需要注意 micro_batch_size 设置
  - 4 卡并行时可用 `accelerate` 或 PyTorch DDP
  - 建议 micro_batch_size=1~2 per GPU，用 gradient accumulation 凑大 batch
  - vLLM 推理时可用 `tensor_parallel_size=4` 加速 rollout 生成

## Development Workflow

```bash
# Install dependencies
uv sync --no-install-package flash-attn && uv sync

# Run all tests
uv run pytest

# Run specific test
uv run pytest -k test_tokenize_prompt_and_output

# Run tests with verbose output
uv run pytest -v

# Generate submission
bash test_and_make_submission.sh
```

## Implementation Guide

All functions to implement are in `tests/adapters.py`. Each raises `NotImplementedError` by default. The actual implementations should go in `cs336_alignment/` and be imported by the adapters.

### Implementation Order (recommended)

**Phase 1 - SFT foundations:**
1. `run_tokenize_prompt_and_output` - tokenize prompt+output, build response_mask
2. `run_compute_entropy` - entropy of logits distribution
3. `run_get_response_log_probs` - forward pass → per-token log probs
4. `run_masked_normalize` - sum and normalize with mask
5. `run_sft_microbatch_train_step` - SFT loss + backward for one microbatch

**Phase 2 - GRPO:**
6. `run_compute_group_normalized_rewards` - group-level advantage computation
7. `run_compute_naive_policy_gradient_loss` - basic REINFORCE loss
8. `run_compute_grpo_clip_loss` - PPO-style clipped loss
9. `run_compute_policy_gradient_loss` - dispatch wrapper
10. `run_masked_mean` - mean with mask
11. `run_grpo_microbatch_train_step` - GRPO loss + backward for one microbatch

**Phase 3 - Optional (RLHF/Safety):**
12. `get_packed_sft_dataset` - packed SFT dataset for instruction tuning
13. `run_iterate_batches` - batch iterator
14. `run_parse_mmlu_response` / `run_parse_gsm8k_response` - output parsing
15. `run_compute_per_instance_dpo_loss` - DPO loss

## Testing Details

- Tests use **snapshot testing** with `.npz` files in `tests/_snapshots/`
- Tolerances: `rtol=1e-4, atol=1e-2` (not exact match by default)
- Use `--snapshot-exact` flag for exact matching
- Fixtures are defined in `tests/conftest.py` with fixed random seeds (`torch.manual_seed(42)`)

## Important Patterns

- **Shifted labels**: `labels = input_ids[:, 1:]` with padding appended
- **Response mask**: 1 for response tokens in `labels`, 0 for prompt/padding
- **Gradient accumulation**: loss must be divided by `gradient_accumulation_steps` before `.backward()`
- **GRPO advantage**: computed per-group (G responses per prompt), normalized by group mean/std
- **Clip ratio**: `ratio = exp(log_prob - old_log_prob)`, clipped to `[1-eps, 1+eps]`

## Coding Conventions

- Type hints used throughout (`from __future__ import annotations`)
- PyTorch tensors for all numerical operations
- HuggingFace `transformers` for tokenizer and model
- `vllm` for efficient inference/rollout generation
- `wandb` for experiment tracking

## GSM8K Dataset Notes

- GSM8K 是小学数学应用题数据集，比 MATH 更简单
- 训练集 ~7.5K 题，测试集 ~1.3K 题
- 答案格式：最终数字答案（通常在 `#### <answer>` 后）
- 需要适配 reward function 以匹配 GSM8K 的答案格式
- 可以通过 HuggingFace Datasets 获取：`datasets.load_dataset("gsm8k", "main")`

## NSCC Job Submission

```bash
# 典型的 PBS/SLURM job script 结构
#PBS -l select=1:ngpus=4
#PBS -l walltime=04:00:00
#PBS -q normal

# 或 SLURM
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
```

## Latest Evaluation Status (2026-02-14)

- **Current baseline evaluation** has completed on GSM8K test split using `cs336_alignment/math_baseline.py`.
- **Result summary** (from `test_log.json`, n=1319):
  - accuracy: `0.0348749052` (~3.49%)
  - format_reward: `0.2934040940` (~29.34%)
  - type1/type2/type3: `46 / 341 / 932`
- **Output file**: `/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B/test_log.json`

## Evaluation Notes (Important)

- Use this command in project root:
  - `uv run python -m cs336_alignment.math_baseline`
- Do **not** use `uv python cs336_alignment/math_baseline.py` (invalid `uv` subcommand usage).
- `math_baseline.py` expects `data/gsm8k/test.jsonl` for evaluation input.
- In NSCC PBS jobs, `CUDA_VISIBLE_DEVICES` may be UUID-based (`GPU-...`). The script has been updated to normalize UUIDs to device indices before creating vLLM `LLM(...)`.
- `drgrpo_grader.py` may print `SyntaxWarning` for regex escape sequences; these warnings are non-fatal for current evaluation runs.

## Latest SFT Experiment Status (2026-02-16)

- **SFT training run** has completed with periodic vLLM evaluation on GSM8K test (`n=1319`).
- **Eval cadence**: every `1000` steps, checkpoints saved under `sft_eval/{step}` (and mirrored under `cs336_alignment/sft_eval/{step}`).
- **Final metrics** (step `22000`):
  - accuracy: `0.3282789992` (~32.83%)
  - type1/type2/type3: `433 / 606 / 280`
  - train/loss: `1.9609375`
  - train/grad_norm: `138`
- **Best checkpoint**: step `18000`, accuracy `0.3305534496` (~33.06%).
- **Outcome**: exceeds assignment target of at least `15%` validation accuracy on full dataset.

## W&B / Dependency Notes (2026-02-16)

- Cluster network/proxy may block online sync; current workflow uses **W&B offline** logs.
- Offline run directory:
  - `wandb/offline-run-20260216_001131-259v26t8`
- `wandb` dependency is pinned to:
  - `wandb==0.24.2` in `pyproject.toml`
- After dependency edits, run:
  - `uv lock`
  - `uv sync --no-install-package flash-attn`
