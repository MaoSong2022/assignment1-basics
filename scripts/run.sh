#!/bin/bash

# 1. Resource Requests
#SBATCH --job-name=cs336_assignment1       # Job name
#SBATCH --partition=DataFrontier_Knowledge
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for python)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --gres=gpu:1                   # number of gpus

# 2. Logging and Notifications
#SBATCH --output=logs/%x_%j.out        # %x=job-name, %j=job-id
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maosong@pjlab.org.cn


# Environment Setup
[ -f .env ] && source .env

echo $https_proxy
echo $TRAIN_PATH
echo $VAL_PATH

# Execution
echo "Starting job at $(date)"
python cs336_basics/train/train.py \
    --project_name "cs336_assignment1" \
    --run_name "SiLU_activation" \
    --train_file_path "$TRAIN_PATH" \
    --valid_file_path "$VAL_PATH" \
    --batch_size 512 \
    --cosine_cycle_iters 4000 \
    --eval_interval_steps 500 \
    --max_learning_rate 1e-2 \
    --epochs 1 \
    --num_workers 4 \
    --d_model 512 \
    --num_heads 16 \
    --betas 0.9 0.95
echo "Finished at $(date)"