#!/bin/bash
# GRPO 训练脚本 — 在交互式 GPU 节点上运行
# 用法：
#   qsub -I -l select=1:ngpus=2:mem=128gb -l walltime=08:00:00
#   cd /scratch/users/nus/e1553316/assignment5
#   bash run_grpo.sh

set -euo pipefail

export ASSIGNMENT5_MODEL_ID=/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B
export HF_HOME=/scratch/users/nus/e1553316/.cache/huggingface
export VLLM_LOGGING_LEVEL=WARNING

cd /scratch/users/nus/e1553316/assignment5
source .venv/bin/activate

# ── 数据 / 模型路径 ──
PATH_ARGS=(
    --train-path       data/math/train.jsonl
    --validation-path  data/math/validation.jsonl
    --model-path       /scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B
    --output-dir       outputs/grpo/run1_onpolicy
)

# ── 训练超参 ──
# train_batch_size=256, gradient_accumulation_steps=128
# → micro_train_batch_size = 256 // 128 = 2
# epochs_per_rollout_batch=1，on-policy 模式，使用 reinforce_with_baseline

TRAIN_ARGS=(
    --seed                        42
    --n-grpo-steps                256
    --rollout-batch-size          256
    --group-size                  8
    --gradient-accumulation-steps 128
    --train-batch-size            256
    --epochs-per-rollout-batch    1
    --loss-type                   reinforce_with_baseline
    --lr                          1e-5
    --advantage-eps               1e-6
    --use-std-normalization
)

# ── vLLM / rollout 采样 ──
ROLLOUT_ARGS=(
    --vllm-device              cuda:1
    --gpu-memory-utilization   0.85
    --rollout-temperature      1.0
    --rollout-top-p            1.0
    --sampling-min-tokens      4
    --sampling-max-tokens      1024
)

# ── 验证 ──
EVAL_ARGS=(
    --eval-every-steps   10
    --max-eval-samples   1024
    --eval-temperature   1.0
    --eval-top-p         1.0
    --eval-max-tokens    1024
)

# ── W&B（不传 --wandb-project 则不启用）──
WANDB_ARGS=(
    --wandb-project  cs336-a5
    --wandb-group    grpo
    --wandb-run-name run1_onpolicy
)

uv run python -m cs336_alignment.grpo_train_loop \
    "${PATH_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
