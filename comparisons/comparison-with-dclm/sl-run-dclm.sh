#!/bin/bash

#SBATCH --job-name=dclm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --account=project_462000883
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module use module use /appl/local/csc/modulefiles/
module load pytorch/2.5
# this should include fasttext (0.9.3)
export PYTHONPATH=/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/comparisons/comparison-with-dclm/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH

export HF_HOME=/scratch/project_462000883/hf_cache

for register in "HI-IN" "HI" "IN" "ID" "IP" "LY" "NA" "MT" "OP" "LY" "SP" "ne" "dtp"; do 
    srun python dclm.py < /scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/comparisons/comparison-with-edu/result/${register}_with_edu.jsonl  > result/${register}_with_edu_and_dclm.jsonl
done
