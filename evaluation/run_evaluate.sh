#!/bin/bash


REGISTER=$1
evaluation="${2:-fineweb}"

ckpts="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints_converted/${REGISTER}"

for dir in $ckpts/*; do
    iteration_number=$(echo $dir | rev | cut -f 1 -d"_" | rev )
    already_evaluated=$(find eval_results/${evaluation}/${REGISTER}/${REGISTER}_iter_${iteration_number} -maxdepth 1 -type f -name "*.json" |wc -l)
    if (( already_evaluated >= 1 )); then
        echo "Directory ${REGISTER}_iter_${iteration_number} exists and contains results, no recalculation."
        continue 1
    fi
    sbatch evaluate.sh $evaluation $REGISTER $iteration_number 
    #sleep 2
done