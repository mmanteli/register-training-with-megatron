#!/bin/bash

model=$1
evaluation="${2:-fineweb}"

case $model in
    "fineweb")
        model_prefix="fineweb"
    ;;
    "hplt-v2-dedup")
        model_prefix="dedup"
    ;;
    *)
        echo "give model name as parametre"
    ;;
esac

# this added so that the jobs do not start simultaneously; small-g usually launches all jobs at the same time
sleep_overlap=0
for iteration_number in `seq -w 0001000 1000 0050000`; do
    # check if we have already evaluated this
    if [ -d "eval_results/${evaluation}/${model}/${model_prefix}_iter_${iteration_number}" ]; then
        already_evaluated=$(find eval_results/${evaluation}/${model}/${model_prefix}_iter_${iteration_number} -maxdepth 1 -type f -name "*.json" |wc -l)
    else
        already_evaluated=0
    fi
    
    if (( already_evaluated >= 1 )); then
        echo "Directory ${model}_iter_${iteration_number} exists and contains results, no recalculation."
        continue 1
    fi
    sbatch evaluate_baselines.sh $evaluation $model $iteration_number $sleep_overlap
    sleep_overlap=$(($sleep_overlap + 20))
done