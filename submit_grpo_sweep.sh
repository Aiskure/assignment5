#!/bin/bash
# 一键提交 GRPO LR sweep 的两个 job
# 用法：bash submit_grpo_sweep.sh

set -euo pipefail

cd /scratch/users/nus/e1553316/assignment5

JOB1=$(qsub -v RUN_NAME=run4_lr2e5,LR=2e-5   run_grpo_sweep.pbs)
JOB2=$(qsub -v RUN_NAME=run5_lr1p5e5,LR=1.5e-5 run_grpo_sweep.pbs)

echo "Submitted: $JOB1  (run4_lr2e5,   lr=2e-5)"
echo "Submitted: $JOB2  (run5_lr1p5e5, lr=1.5e-5)"
echo ""
qstat -u e1553316
