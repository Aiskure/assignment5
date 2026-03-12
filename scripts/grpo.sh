#!/usr/bin/env bash
# GRPO 训练脚本
#
# 用法：
#   bash scripts/grpo.sh
#
# 可通过环境变量覆盖关键参数：
#   ASSIGNMENT5_MODEL_ID=<model_path> \
#   RUN_NAME=my_run \
#   LR=1e-5 \
#   LOSS_TYPE=grpo_clip \
#   KL_COEF=3e-4 \
#   EPOCHS=2 \
#   bash scripts/grpo.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

export ASSIGNMENT5_MODEL_ID="${ASSIGNMENT5_MODEL_ID:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# ── 可覆盖的超参 ──
RUN_NAME="${RUN_NAME:-run_grpo}"
LR="${LR:-1.5e-5}"
LOSS_TYPE="${LOSS_TYPE:-grpo_clip}"
KL_COEF="${KL_COEF:-3e-4}"
EPOCHS="${EPOCHS:-2}"
TRAIN_BS="${TRAIN_BS:-256}"
GRAD_ACC="${GRAD_ACC:-128}"
USE_STD="${USE_STD:-true}"
GROUP_FILTER="${GROUP_FILTER:-true}"
N_STEPS="${N_STEPS:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo/${RUN_NAME}}"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "[GRPO] start  $(date)"
echo "[GRPO] run=$RUN_NAME  lr=$LR  loss=$LOSS_TYPE  kl=$KL_COEF  epochs=$EPOCHS"
echo "[GRPO] output=$OUTPUT_DIR"
echo "========================================="

# ── use_std 的 flag ──
if [ "$USE_STD" = "true" ]; then
    STD_FLAG="--use-std-normalization"
else
    STD_FLAG="--no-use-std-normalization"
fi

if [ "$GROUP_FILTER" = "true" ]; then
    FILTER_FLAG="--group-filter"
else
    FILTER_FLAG="--no-group-filter"
fi

PATH_ARGS=(
    --train-path       data/math/train.jsonl
    --validation-path  data/math/validation.jsonl
    --model-path       "$ASSIGNMENT5_MODEL_ID"
    --output-dir       "$OUTPUT_DIR"
)

TRAIN_ARGS=(
    --seed                        42
    --n-grpo-steps                "$N_STEPS"
    --rollout-batch-size          "$TRAIN_BS"
    --train-batch-size            "$TRAIN_BS"
    --group-size                  8
    --gradient-accumulation-steps "$GRAD_ACC"
    --epochs-per-rollout-batch    "$EPOCHS"
    --loss-type                   "$LOSS_TYPE"
    --loss-aggregation            masked_normalize
    --lr                          "$LR"
    --kl-coef                     "$KL_COEF"
    --advantage-eps               1e-6
    $STD_FLAG
    $FILTER_FLAG
)

ROLLOUT_ARGS=(
    --vllm-device              cuda:1
    --gpu-memory-utilization   0.85
    --rollout-temperature      1.0
    --rollout-top-p            1.0
    --sampling-min-tokens      4
    --sampling-max-tokens      1024
)

EVAL_ARGS=(
    --eval-every-steps   10
    --max-eval-samples   1024
    --eval-temperature   1.0
    --eval-top-p         1.0
    --eval-max-tokens    1024
)

WANDB_ARGS=(
    --wandb-project  cs336-a5
    --wandb-group    grpo
    --wandb-run-name "$RUN_NAME"
)

uv run python -m cs336_alignment.grpo_train_loop \
    "${PATH_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "========================================="
echo "[GRPO] finished  $(date)"
echo "========================================="
