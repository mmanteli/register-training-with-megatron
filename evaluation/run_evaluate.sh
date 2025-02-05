#!/bin/bash


REGISTER=$1
evaluation="${2:-fineweb}"

ckpts="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints_converted/${REGISTER}"

for dir in $ckpts/*; do
    iteration_number=$(echo $dir | rev | cut -f 1 -d"_" | rev )
    sbatch evaluate.sh $evaluation $REGISTER $iteration_number 
done