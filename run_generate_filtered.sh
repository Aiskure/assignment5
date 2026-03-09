#!/bin/bash
# 生成 rejection-sampled 过滤数据集，再用 sft.py 在过滤后的数据上训练
# 用法（交互式节点）：
#   qsub -I -l select=1:ngpus=2:mem=128gb -l walltime=04:00:00
#   cd /scratch/users/nus/e1553316/assignment5
#   bash run_generate_filtered.sh

set -euo pipefail

export ASSIGNMENT5_MODEL_ID=/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B
export HF_HOME=/scratch/users/nus/e1553316/.cache/huggingface
export VLLM_LOGGING_LEVEL=WARNING
export WANDB_MODE=offline

cd /scratch/users/nus/e1553316/assignment5
source .venv/bin/activate

echo "========================================="
echo "[Step 1] 生成过滤数据集  $(date)"
echo "========================================="

uv run python -m cs336_alignment.generate_filtered_sft_data \
    --model-path   "$ASSIGNMENT5_MODEL_ID" \
    --train-path   data/math/train.jsonl \
    --output-path  data/math/train_filtered.jsonl \
    --rollouts-per-problem 4 \
    --temperature  1.0 \
    --top-p        1.0 \
    --max-tokens   1024 \
    --vllm-device  cuda:1 \
    --gpu-memory-utilization 0.85 \
    --seed         42

echo ""
echo "========================================="
echo "[Step 2] 在过滤数据集上训练 SFT  $(date)"
echo "========================================="

uv run python -m cs336_alignment.sft \
    --model-path   "$ASSIGNMENT5_MODEL_ID" \
    --train-path   data/math/train_filtered.jsonl \
    --log-directory outputs/sft/run_filtered \
    --max-train-examples 0 \
    --seed             42 \
    --epochs           3 \
    --micro-batch-size 1 \
    --local-batch-size 32 \
    --lr               1e-5 \
    --log-every-local-steps      10 \
    --eval-every-local-steps     0 \
    --run-baseline-eval-at-step0 \
    --run-final-eval \
    --vllm-device               cuda:1 \
    --vllm-gpu-memory-utilization 0.85 \
    --save-final-checkpoint \
    --save-every-local-steps 0 \
    --use-wandb \
    --wandb-project cs336-a5-sft \
    --wandb-run-name run_filtered

echo "========================================="
echo "[DONE] Filtered SFT experiment finished.  $(date)"
echo "========================================="
