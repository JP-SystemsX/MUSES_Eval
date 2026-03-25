#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # GPU Count
#SBATCH --nodes=1
#SBATCH --mem=140G # Working Memory
#SBATCH --time=72:00:00  # Should take at most 20h but in rare exceptions might take longer 
#SBATCH --account=p_llm_timeseries
#SBATCH --job-name=DUESE
#SBATCH --output=hpc/logs/DUESE-%j-%a.out  # Output Address 
#SBATCH --error=hpc/logs/DUESE-%j-%a.err  # Output Address
#SBATCH --array=0-128%16

# Derive parameters
DATA_ID=$(( SLURM_ARRAY_TASK_ID / 8 ))           # number between 0–35
MODEL_ID=$(( SLURM_ARRAY_TASK_ID % 8 ))    # index into file list

source ./hpc/modules.sh
srun python3 ./src/main.py --data-id $DATA_ID --model-id $MODEL_ID --trial-count 150