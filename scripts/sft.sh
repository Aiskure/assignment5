#!/usr/bin/env bash
# SFT 训练脚本
#
# 用法：
#   bash scripts/sft.sh
#
# 可通过环境变量覆盖关键参数：
#   ASSIGNMENT5_MODEL_ID=<model_path> \
#   RUN_NAME=my_run \
#   N=0 \          # 0=全量，其他=截断条数
#   LR=1e-5 \
#   EPOCHS=3 \
#   bash scripts/sft.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

export ASSIGNMENT5_MODEL_ID="${ASSIGNMENT5_MODEL_ID:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── 可覆盖的超参 ──
N="${N:-0}"          # 0 = 全量训练集
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-3}"
MICRO_BS="${MICRO_BS:-1}"
LOCAL_BS="${LOCAL_BS:-32}"

if [ "$N" -eq 0 ]; then
    RUN_NAME="${RUN_NAME:-run_sft_full}"
else
    RUN_NAME="${RUN_NAME:-run_sft_n${N}}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft/${RUN_NAME}}"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "[SFT] start  $(date)"
echo "[SFT] run=$RUN_NAME  lr=$LR  epochs=$EPOCHS  N=$N"
echo "[SFT] output=$OUTPUT_DIR"
echo "========================================="

PATH_ARGS=(
    --model-path     "$ASSIGNMENT5_MODEL_ID"
    --train-path     data/math/train.jsonl
    --log-directory  "$OUTPUT_DIR"
    --max-train-examples "$N"
)

TRAIN_ARGS=(
    --seed             42
    --epochs           "$EPOCHS"
    --micro-batch-size "$MICRO_BS"
    --local-batch-size "$LOCAL_BS"
    --lr               "$LR"
)

EVAL_ARGS=(
    --log-every-local-steps      10
    --eval-every-local-steps     0
    --run-baseline-eval-at-step0
    --run-final-eval
)

VLLM_ARGS=(
    --vllm-device                 cuda:1
    --vllm-gpu-memory-utilization 0.85
)

CKPT_ARGS=(
    --save-final-checkpoint
    --save-every-local-steps 0
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project cs336-a5-sft
    --wandb-run-name "$RUN_NAME"
)

uv run python -m cs336_alignment.sft \
    "${PATH_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${VLLM_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "========================================="
echo "[SFT] finished  $(date)"
echo "========================================="
