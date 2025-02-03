#!/bin/bash
#SBATCH --job-name=convert_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=50G
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --account=project_462000615
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# this contains dification for output head (related to 1.71B vs 1.82B difference)
register=$1

if [ "$#" -ne 1 ]; then
    echo "Usage: input register as parameter"
    exit 1
fi


echo "CONVERTING ${REGISTER} CHECKPOINTS"

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

    b=$(basename $checkpoint)
    output="${o}/${register}/${b}"
    #echo $output
    mkdir -p $output

    # see if we have already converted this
    FILE_COUNT=$(find "$output" -maxdepth 1 -type f -name "*.safetensors" | wc -l)
    if (( FILE_COUNT >= 2 )); then
        echo "Directory exists and contains at least two .safetensors files, no recalculation."
        continue 1
    fi
    srun python3 ${eval_path}/tools/checkpoint/util.py \
                    --model-type GPT \
                    --loader loader_megatron \
                    --saver saver_llama2_hf \
                    --tokenizer-dir gpt2-tokenizer \
                    --load-dir "$checkpoint" \
                    --save-dir "$output"

    echo "FIRST DONE; EXITING"
    exit
done

#srun python3 ${eval_path}/tools/checkpoint/util.py \
#                    --model-type GPT \
#                    --loader loader_megatron \
#                    --saver saver_llama2_hf \
#                    --tokenizer-dir gpt2-tokenizer \
#                    --load-dir "$ckpt" \
#                    --save-dir "$o"