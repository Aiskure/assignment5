#!/usr/bin/env bash
# 一键提交所有 EI 控制变量实验（6 个 PBS job，可并行运行）
#
# 用法：
#   bash submit_ei_all.sh
#
# 实验设计（控制变量法）：
#   baseline:   G=4, D_b=1024, epoch=1
#   vary G:     G=8, D_b=1024, epoch=1
#   vary D_b:   G=4, D_b=512,  epoch=1
#               G=4, D_b=2048, epoch=1
#   vary epoch: G=4, D_b=1024, epoch=2
#               G=8, D_b=1024, epoch=2

set -euo pipefail

PROJECT=/scratch/users/nus/e1553316/assignment5
PBS_SCRIPT=$PROJECT/run_ei.pbs
LOG_BASE=$PROJECT/outputs/ei

submit() {
  local G=$1 DB=$2 EPOCHS=$3
  local RUN_NAME="ei_G${G}_Db${DB}_E${EPOCHS}"
  local LOG_DIR="$LOG_BASE/$RUN_NAME"
  mkdir -p "$LOG_DIR"
  qsub \
    -N "ei_${G}_${DB}_${EPOCHS}" \
    -o "$LOG_DIR/pbs.log" \
    -v "EI_G=${G},EI_DB=${DB},EI_EPOCHS=${EPOCHS},RUN_NAME=${RUN_NAME}" \
    "$PBS_SCRIPT"
  echo "submitted: $RUN_NAME"
}

# baseline
submit 4 1024 1

# vary G
submit 8 1024 1

# vary D_b
submit 4 512  1
submit 4 2048 1

# vary epoch
submit 4 1024 2
submit 8 1024 2
