#!/bin/bash
#SBATCH --job-name=nvidia
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --mem=25G
#SBATCH --partition=small-g
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --account=project_462000883
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module use module use /appl/local/csc/modulefiles/
module load pytorch/2.5
register=$1
export HF_HOME=/scratch/project_462000883/hf_cache

srun python nvidia.py < /scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/comparisons/sample/${register}.jsonl  > result/${register}_with_nvidia.jsonl
