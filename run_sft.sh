#!/bin/bash
# SFT 训练脚本 — 在交互式 GPU 节点上运行
# 用法：
#   qsub -I -l select=1:ngpus=2:mem=128gb -l walltime=08:00:00
#   cd /scratch/users/nus/e1553316/assignment5
#   bash run_sft.sh              # 运行全量数据集
#   N=128 bash run_sft.sh        # 运行 128 条子集

set -euo pipefail

export ASSIGNMENT5_MODEL_ID=/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B
export HF_HOME=/scratch/users/nus/e1553316/.cache/huggingface
export VLLM_LOGGING_LEVEL=WARNING
export WANDB_MODE=offline

cd /scratch/users/nus/e1553316/assignment5
source .venv/bin/activate

# 训练集规模：0 = 全量，其余为截断条数
N="${N:-0}"

if [ "$N" -eq 0 ]; then
    RUN_NAME="run_full"
else
    RUN_NAME="run_n${N}"
fi

uv run python -m cs336_alignment.sft \
    \
    `# ── 数据 / 模型路径 ──` \
    --model-path   /scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B \
    --train-path   data/math/train.jsonl \
    --log-directory outputs/sft/${RUN_NAME} \
    --max-train-examples ${N} \
    \
    `# ── 训练超参 ──` \
    --seed             42 \
    --epochs           3 \
    --micro-batch-size 1 \
    --local-batch-size 32 \
    --lr               1e-5 \
    \
    `# ── 日志 / 评估 ──` \
    --log-every-local-steps      10 \
    --eval-every-local-steps     0 \
    --run-baseline-eval-at-step0 \
    --run-final-eval \
    \
    `# ── vLLM ──` \
    --vllm-device               cuda:1 \
    --vllm-gpu-memory-utilization 0.85 \
    \
    `# ── checkpoint ──` \
    --save-final-checkpoint \
    --save-every-local-steps 0 \
    \
    `# ── W&B ──` \
    --use-wandb \
    --wandb-project cs336-a5-sft \
    --wandb-run-name ${RUN_NAME}
