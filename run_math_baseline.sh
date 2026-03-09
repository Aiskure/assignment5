#!/usr/bin/env bash
#qsub -I -l select=1:ngpus=1:mem=64gb -l walltime=02:00:00
set -euo pipefail

PROJECT_ROOT="/home/users/nus/e1553316/scratch/assignment5"
cd "$PROJECT_ROOT"

export ASSIGNMENT5_MODEL_ID="${ASSIGNMENT5_MODEL_ID:-/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B}"
export HF_HOME="${HF_HOME:-/scratch/users/nus/e1553316/.cache/huggingface}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

DATASET_PATH="${DATASET_PATH:-data/math/validation.jsonl}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1024}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-seeded_random}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baseline/math_val_${MAX_EVAL_SAMPLES}_seed${SAMPLE_SEED}}"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Running math baseline"
echo "[INFO] model=$ASSIGNMENT5_MODEL_ID"
echo "[INFO] dataset=$DATASET_PATH max_eval_samples=$MAX_EVAL_SAMPLES strategy=$SAMPLING_STRATEGY seed=$SAMPLE_SEED"
echo "[INFO] output_dir=$OUTPUT_DIR"

uv run python -m cs336_alignment.math_baseline \
    --model-path        "$ASSIGNMENT5_MODEL_ID" \
    --dataset-path      "$DATASET_PATH" \
    --max-eval-samples  "$MAX_EVAL_SAMPLES" \
    --sampling-strategy "$SAMPLING_STRATEGY" \
    --sample-seed       "$SAMPLE_SEED" \
    --output-dir        "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"
