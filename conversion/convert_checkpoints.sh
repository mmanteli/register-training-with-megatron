#!/bin/bash
#SBATCH --job-name=convert_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=50G
#SBATCH --partition=small
#SBATCH --time=01:29:00
#SBATCH --account=project_462000615
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


register=$1
limit="0015000"  # limiting to this for now

if [ "$#" -ne 1 ]; then
    echo "Usage: input register as parameter"
    exit 1
fi


echo "CONVERTING ${register} CHECKPOINTS"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

# this version of the code contains minor change that facilitates model output head
eval_path="/scratch/project_462000353/pyysalos/second-hplt-eval/Megatron-LM-lumi" 

# input and output
ckpt="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints"
o=${ckpt/"checkpoints"/"checkpoints_converted"}

for checkpoint in $ckpt/$register/*; do

    # skip latest.txt file
    if [[ -f $checkpoint ]]; then
        continue
    fi

    # skip if iteration number is over the limit:
    iteration=$(echo $checkpoint | rev | cut -f 1 -d"_" | rev )  # iteration number is the last field separated by _
    if (( 10#$iteration > 10#$limit )); then
        echo "skipping ${iteration} due to limit"
        continue 1
    fi

    b=$(basename $checkpoint)
    output="${o}/${register}/${b}"
    #echo $output
    mkdir -p $output

    # see if we have already converted this
    FILE_COUNT=$(find "$output" -maxdepth 1 -type f -name "*model.bin" | wc -l)
    if (( FILE_COUNT >= 1 )); then
        echo "Directory iter_${iteration} exists and contains model.bin, no recalculation."
        continue 1
    fi
    srun python3 ${eval_path}/tools/checkpoint/util.py \
                    --model-type GPT \
                    --loader loader_megatron \
                    --saver saver_llama2_hf \
                    --tokenizer-dir gpt2-tokenizer/ \
                    --load-dir "$checkpoint" \
                    --save-dir "$output"
    echo " "
    echo "-----------------------------------------------------------------"
    echo " "

done

mkdir -p logs/done
cp logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err logs/done/${register}_${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err
cp logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out logs/done/${register}_${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out