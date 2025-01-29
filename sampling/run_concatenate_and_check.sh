#!/bin/bash
#SBATCH --job-name=concatenate
#SBATCH --account=project_462000615
#SBATCH --partition=small
#SBATCH --time=10:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=64
#SBATCH -o logs/%j.out

module load LUMI
module load parallel

REGISTER=IN
lang="eng_Latn"
suffix=""

echo "CONCATENATE REGISTER ${REGISTER}"

#data="/scratch/project_462000353/amanda/register-training/register-model-training/sampling/results/${lang}/${REGISTER}"
data="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/sampling/results/${lang}/${REGISTER}"
output="/scratch/project_462000353/HPLT-REGISTERS/samples-150B-by-register-xlmrl/original_corrected"

mkdir -p $output

echo "Start: $(date)"

cat ${data}/*[0-9].jsonl | parallel --pipe -j64 python3 concatenate_and_check.py > ${output}/${lang}_${REGISTER}${suffix}.jsonl

echo "end: $(date)"

cp logs/$SLURM_JOBID.out logs/concat-${REGISTER}.out
