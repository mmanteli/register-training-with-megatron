#!/bin/bash
#SBATCH --job-name=concatenate_HI
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=8:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=64
#SBATCH -o logs/%j.out

module load LUMI
module load parallel

REGISTER=$1
lang="eng_Latn"

echo "CONCATENATE REGISTER ${REGISTER}"

data="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/sampling/results/${lang}/${REGISTER}"
output="/scratch/project_462000353/HPLT-REGISTERS/samples-150B-by-register-xlmrl/original_corrected"

mkdir -p $output

echo "Start: $(date)"

cat ${data}/*[0-9].jsonl | parallel --pipe -j64 python3 concatenate_and_check.py > ${output}/${lang}_${REGISTER}_with_th_1.jsonl

echo "end: $(date)"

cp logs/$SLURM_JOBID.out logs/concat-${REGISTER}.out
