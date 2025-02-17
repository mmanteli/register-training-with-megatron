#!/bin/bash


REGISTER=$1
evaluation="${2:-fineweb}"

ckpts="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints_converted/${REGISTER}"
reg_truncated=$(echo $REGISTER | rev | cut -f1 -d"_" | rev)  # this is for previous naming error; e.g. HI-IN_HI_dtp was accidentally saved as "dtp" only
# for un-hyphened register works the same as before

# this added so that the jobs do not start simultaneously; small-g usually launches all jobs at the same time
sleep_overlap=0
for dir in $ckpts/*; do
    iteration_number=$(echo $dir | rev | cut -f 1 -d"_" | rev )
    # check if we have already evaluated this
    if [ -d "eval_results/${evaluation}/${REGISTER}/${reg_truncated}_iter_${iteration_number}" ]; then
        already_evaluated=$(find eval_results/${evaluation}/${REGISTER}/${reg_truncated}_iter_${iteration_number} -maxdepth 1 -type f -name "*.json" |wc -l)
    else
        already_evaluated=0
    fi
    
    if (( already_evaluated >= 1 )); then
        echo "Directory ${REGISTER}_iter_${iteration_number} exists and contains results, no recalculation."
        continue 1
    fi
    sbatch evaluate.sh $evaluation $REGISTER $iteration_number $sleep_overlap
    sleep_overlap=$(($sleep_overlap + 20))
done