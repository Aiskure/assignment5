#!/usr/bin/env bash
# Expert Iteration 训练脚本
#
# 用法：
#   bash scripts/ei.sh
#
# 可通过环境变量覆盖关键参数：
#   ASSIGNMENT5_MODEL_ID=<model_path> \
#   RUN_NAME=my_run \
#   N_EI_STEPS=5 \
#   DB_SIZE=1024 \
#   LR=1e-5 \
#   bash scripts/ei.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

export ASSIGNMENT5_MODEL_ID="${ASSIGNMENT5_MODEL_ID:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── 可覆盖的超参 ──
RUN_NAME="${RUN_NAME:-run_ei}"
N_EI_STEPS="${N_EI_STEPS:-5}"
DB_SIZE="${DB_SIZE:-1024}"
ROLLOUTS="${ROLLOUTS:-8}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
LR="${LR:-1e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ei/${RUN_NAME}}"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "[EI] start  $(date)"
echo "[EI] run=$RUN_NAME  steps=$N_EI_STEPS  db=$DB_SIZE  rollouts=$ROLLOUTS  lr=$LR"
echo "[EI] output=$OUTPUT_DIR"
echo "========================================="

MODEL_ARGS=(
    --model-path       "$ASSIGNMENT5_MODEL_ID"
    --train-path       data/math/train.jsonl
    --validation-path  data/math/validation.jsonl
    --output-dir       "$OUTPUT_DIR"
)

EI_ARGS=(
    --seed                42
    --n-ei-steps          "$N_EI_STEPS"
    --db-size             "$DB_SIZE"
    --rollouts-per-question "$ROLLOUTS"
    --sft-epochs-per-ei   "$SFT_EPOCHS"
)

TRAIN_ARGS=(
    --micro-batch-size 1
    --local-batch-size 32
    --lr               "$LR"
)

ROLLOUT_ARGS=(
    --vllm-device              cuda:1
    --gpu-memory-utilization   0.85
    --rollout-temperature      1.0
    --rollout-top-p            1.0
    --rollout-max-tokens       1024
)

EVAL_ARGS=(
    --max-eval-samples  1024
    --eval-temperature  1.0
    --eval-top-p        1.0
    --eval-max-tokens   1024
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project  cs336-a5
    --wandb-group    ei
    --wandb-run-name "$RUN_NAME"
)

uv run python -m cs336_alignment.expert_iteration_experiment \
    "${MODEL_ARGS[@]}" \
    "${EI_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "========================================="
echo "[EI] finished  $(date)"
echo "========================================="
