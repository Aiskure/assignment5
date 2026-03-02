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

**Important dataset note**:
- **Current working dataset (GRPO / EI)**: local **MATH-format** dataset at `data/math/*.jsonl`.
- **GSM8K** is kept for baseline/SFT historical evaluation and comparison scripts.

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

- **Model**: Qwen 2.5 Math 1.5B (set via `ASSIGNMENT5_MODEL_ID`; AutoDL default: `/root/autodl-tmp/models/Qwen2.5-Math-1.5B`)
- **Dataset**:
  - `data/math/*.jsonl` for current GRPO train loop and EI experiments
  - `data/gsm8k/*.jsonl` for baseline / SFT historical comparison runs
- **Package manager**: `uv` (not pip)
- **Python**: 3.11 or 3.12 (not 3.13)
- **Current focus**: GRPO train loop + Expert Iteration on MATH-format data

## Current Environment (AutoDL, 2026-02-20)

- **平台**: AutoDL (中国)
- **GPU**: 2x NVIDIA A800 80GB PCIe
- **总显存**: 160GB
- **CPU / RAM**: 28 cores / 200 GB
- **项目路径**: `/root/assignment5`
- **数据盘**: `/root/autodl-tmp`（模型与缓存放这里，避免系统盘占满）
- **注意事项**:
  - 推荐设置 `ASSIGNMENT5_MODEL_ID=/root/autodl-tmp/models/Qwen2.5-Math-1.5B`
  - 推荐使用 `HF_HOME`，不再依赖 `TRANSFORMERS_CACHE`
  - `flash-attn` 已在 CUDA 12.4 环境下编译通过（A800 架构 `sm80`）

## Legacy Environment (NSCC, historical records)

- **集群**: NSCC (National Supercomputing Centre Singapore)
- **GPU**: 4x NVIDIA A100 40GB per job
- **总显存**: 160GB per job
- **说明**: 该环境用于历史实验记录，对当前 AutoDL 运行不再作为默认配置

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

## Latest Evaluation Status (2026-02-20, AutoDL)

- **Current baseline evaluation** has completed on GSM8K test split using `cs336_alignment/math_baseline.py` in AutoDL.
- **Result summary** (from `test_log.json`, n=1319):
  - accuracy: `0.0348749052` (~3.49%)
  - format_reward: `0.2934040940` (~29.34%)
  - type1/type2/type3: `46 / 341 / 932`
- **Output file**: `/root/autodl-tmp/models/Qwen2.5-Math-1.5B/test_log.json`

## Evaluation Notes (Important)

- Use this command in project root:
  - `uv run python -m cs336_alignment.math_baseline`
- Set model path with environment variable:
  - `export ASSIGNMENT5_MODEL_ID=/root/autodl-tmp/models/Qwen2.5-Math-1.5B`
- Do **not** use `uv python cs336_alignment/math_baseline.py` (invalid `uv` subcommand usage).
- `math_baseline.py` expects `data/gsm8k/test.jsonl` for evaluation input.
- In NSCC PBS jobs, `CUDA_VISIBLE_DEVICES` may be UUID-based (`GPU-...`). This is retained as legacy compatibility logic.
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

## Latest EI Experiment Status (2026-02-21, AutoDL)

- **Script**: `cs336_alignment/expert_iteration_experiment.py`
- **Dataset**:
  - train: `data/math/train.jsonl` (11998 rows loaded)
  - eval: `data/math/test.jsonl` with `max_eval_samples=500`
- **Key run config**:
  - `n_ei_steps=5`
  - `db_size=512`
  - `rollouts_per_question=4`
  - `sft_epochs_per_ei=1`
  - `micro_batch_size=1`, `local_batch_size=32`, `lr=1e-5`
  - `vllm_device=cuda:1`
- **Result summary**:
  - step0 eval accuracy: `0.040`
  - step5 eval accuracy: `0.110`
  - step0 -> step5 format reward: `0.244 -> 0.606`
  - status: EI loop runs correctly, but this config is still below the `15%` target.
- **Output directory (data disk)**:
  - `/root/autodl-tmp/assignment5_outputs/cs336_alignment/ei_runs_formal_db512_g4_e1`
- **W&B run**:
  - `https://wandb.ai/762523583-national-university-of-singapore-students-union/cs336-a5-ei/runs/6zze4mni`

## W&B / Dependency Notes (2026-02-16)

- Cluster network/proxy may block online sync; current workflow uses **W&B offline** logs.
- Offline run directory:
  - `wandb/offline-run-20260216_001131-259v26t8`
- `wandb` dependency is pinned to:
  - `wandb==0.24.2` in `pyproject.toml`
- After dependency edits, run:
  - `uv lock`
  - `uv sync --no-install-package flash-attn`
- AutoDL uploads synced from offline runs include:
  - `5ocq572h`
  - `thrmsfms`
  - `6zze4mni`

## Storage Notes (AutoDL)

- System disk (`/`) is small; keep heavy outputs on data disk:
  - `/root/autodl-tmp/assignment5_outputs/`
- Existing EI output symlinks in project:
  - `cs336_alignment/ei_runs_smoke` -> data disk path
  - `cs336_alignment/ei_runs_db512_g4_e1` -> data disk path

## Latest GRPO Adapter Test Status (2026-02-24, AutoDL)

- **Adapter wiring update**:
  - `tests/adapters.py::run_compute_naive_policy_gradient_loss` is now connected to
    `cs336_alignment.utils.compute_naive_policy_gradient_loss`.
  - `run_compute_group_normalized_rewards` remained connected and was re-validated.
- **Targeted pytest command**:
  - `uv run pytest tests/test_grpo.py -k 'compute_group_normalized_rewards or compute_naive_policy_gradient_loss' -q`
- **Result**:
  - `test_compute_group_normalized_rewards_normalize_by_std`: PASSED
  - `test_compute_group_normalized_rewards_no_normalize_by_std`: PASSED
  - `test_compute_naive_policy_gradient_loss`: PASSED
  - Summary: `3 passed, 11 deselected, 1 warning in 2.27s`
- **Warning note**:
  - `TRANSFORMERS_CACHE` deprecation warning from `transformers` is non-blocking.

## Latest GRPO Train Loop Progress (2026-02-27, AutoDL)

- **Primary file under implementation**:
  - `cs336_alignment/grpo_train_loop.py`
- **Current implementation state**:
  - Phase A done: config assertions + derived sizes (`micro_train_batch_size`, prompt/microbatch counts).
  - Phase B done: prompt-level sampling and group expansion for rollout batch construction.
  - Phase C done: vLLM rollout path with policy->vLLM weight sync and response canonicalization.
  - Phase D done: reward/advantage computation via `compute_group_normalized_rewards` with finite checks.
  - Phase E done: rollout-to-training tensorization (`tokenize_prompt_and_output`, response log-prob forward).
  - Phase F done: `old_log_probs` caching for `grpo_clip` using detached tensors.
  - Phase G done (baseline form): microbatch updates, gradient accumulation, grad clipping, and tail-step protection.
- **Data-format support update**:
  - Added `cs336_alignment/data_utils.py::extract_question_and_gt` for unified MATH/GSM8K extraction.
- **Readability/maintenance update**:
  - Added Chinese `step/batch/microbatch` mapping comments in `grpo_train_loop.py`.
- **Known pending items**:
  - Make `epochs_per_rollout_batch > 1` effective (needed for off-policy GRPO experiments).
  - Implement Phase H eval/logging (`val/reward`, `val/format_reward`, `val/answer_reward`, rollout samples).
  - Add `Args + main` entrypoint for cleaner experiment sweeps.
