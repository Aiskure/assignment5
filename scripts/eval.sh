#!/usr/bin/env bash
# 数学基线 / checkpoint 评估脚本
#
# 用法：
#   bash scripts/eval.sh                          # 评估原始模型，1024 条
#
# 可通过环境变量覆盖：
#   ASSIGNMENT5_MODEL_ID=outputs/grpo/run10_ep2_kl3e4/final_checkpoint \
#   MAX_EVAL_SAMPLES=0 \
#   OUTPUT_DIR=outputs/baseline/run10_val5000 \
#   bash scripts/eval.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

export ASSIGNMENT5_MODEL_ID="${ASSIGNMENT5_MODEL_ID:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

DATASET_PATH="${DATASET_PATH:-data/math/validation.jsonl}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1024}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-seeded_random}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baseline/math_val_${MAX_EVAL_SAMPLES}_seed${SAMPLE_SEED}}"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "[EVAL] start  $(date)"
echo "[EVAL] model=$ASSIGNMENT5_MODEL_ID"
echo "[EVAL] dataset=$DATASET_PATH  max_samples=$MAX_EVAL_SAMPLES  strategy=$SAMPLING_STRATEGY  seed=$SAMPLE_SEED"
echo "[EVAL] output=$OUTPUT_DIR"
echo "========================================="

uv run python -m cs336_alignment.evaluate \
    --model-path        "$ASSIGNMENT5_MODEL_ID" \
    --dataset-path      "$DATASET_PATH" \
    --max-eval-samples  "$MAX_EVAL_SAMPLES" \
    --sampling-strategy "$SAMPLING_STRATEGY" \
    --sample-seed       "$SAMPLE_SEED" \
    --output-dir        "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "========================================="
echo "[EVAL] finished  $(date)"
echo "========================================="
