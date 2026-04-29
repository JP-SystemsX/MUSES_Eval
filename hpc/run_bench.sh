#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # GPU Count
#SBATCH --nodes=1
#SBATCH --mem=180G # Working Memory
#SBATCH --time=60:00:00  # Should take at most 24h but in rare exceptions might take longer 
#SBATCH --job-name=DUESE
#SBATCH --output=hpc/logs/DUESE-%j-%a.out  # Output Address 
#SBATCH --error=hpc/logs/DUESE-%j-%a.err  # Output Address
#SBATCH --array=10-80   #TODO 0-89%89

# Derive parameters
DATA_ID=$(( SLURM_ARRAY_TASK_ID % 18 ))     # number between 0–35
MODEL_ID=$(( SLURM_ARRAY_TASK_ID / 18 ))    # index into file list

source ./hpc/modules.sh
srun python3 ./src/main.py --data-id $DATA_ID --model-id $MODEL_ID --trial-count 100