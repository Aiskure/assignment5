#!/usr/bin/env bash
# EI 实验脚本（交互式节点）
# 用法：
#   qsub -I -l select=1:ngpus=2:mem=128gb -l walltime=12:00:00
#   cd /home/users/nus/e1553316/scratch/assignment5
#   bash run_ei.sh
set -euo pipefail

PROJECT_ROOT="/home/users/nus/e1553316/scratch/assignment5"
cd "$PROJECT_ROOT"
source .venv/bin/activate

export ASSIGNMENT5_MODEL_ID="${ASSIGNMENT5_MODEL_ID:-/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B}"
export HF_HOME="${HF_HOME:-/scratch/users/nus/e1553316/.cache/huggingface}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-offline}"

RUN_NAME="${RUN_NAME:-run1_ei}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ei/${RUN_NAME}}"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "[EI] start  $(date)"
echo "[EI] output dir: $OUTPUT_DIR"
echo "========================================="

# 数据 / 模型路径
MODEL_ARGS=(
  --model-path "$ASSIGNMENT5_MODEL_ID"
  --train-path data/math/train.jsonl
  --validation-path data/math/validation.jsonl
  --output-dir "$OUTPUT_DIR"
)

# EI 超参数
EI_ARGS=(
  --seed 42
  --n-ei-steps 5
  --db-size 1024
  --rollouts-per-question 8
  --sft-epochs-per-ei 1
)

# SFT 训练超参数
TRAIN_ARGS=(
  --micro-batch-size 1
  --local-batch-size 32
  --lr 1e-5
)

# rollout 采样超参数
ROLLOUT_ARGS=(
  --vllm-device cuda:1
  --gpu-memory-utilization 0.85
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-max-tokens 1024
)

# 评估超参数
EVAL_ARGS=(
  --max-eval-samples 1024
  --eval-temperature 1.0
  --eval-top-p 1.0
  --eval-max-tokens 1024
)

# 其他
MISC_ARGS=(
  --save-each-step
  --use-wandb
  --wandb-project cs336-a5
  --wandb-group   ei
  --wandb-run-name "$RUN_NAME"
)

uv run python -m cs336_alignment.expert_iteration_experiment \
  "${MODEL_ARGS[@]}" \
  "${EI_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${MISC_ARGS[@]}"

echo "========================================="
echo "[EI] finished  $(date)"
echo "========================================="
