#!/bin/bash
#SBATCH --job-name=evaluate_multiple
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=25G
#SBATCH --partition=small-g
#SBATCH --time=11:58:00
#SBATCH --gres=gpu:mi250:1
#SBATCH --account=project_462000353
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


module purge
module use /appl/local/csc/modulefiles
module load pytorch
export PYTHONPATH=/scratch/project_462000353/amanda/register-training/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH



REGISTER=$1
evaluation="multiple"
echo $REGISTER
export TRANSFORMERS_CACHE="/scratch/project_462000353/cache"
export HF_HOME="/scratch/project_462000353/cache"
DIR="/scratch/project_462000353/amanda/register-training"

for i in `seq 1000 1000 14000`; do
    echo "START ckpt ${i}: $(date)"
    model_to_evaluate="${DIR}/checkpoints_converted/8N/${REGISTER}/global_step${i}"
    srun python ${DIR}/pythonuserbase/bin/lighteval accelerate \
        --model_args "pretrained=${model_to_evaluate},tokenizer=gpt2" \
        --tasks "${DIR}/register-model-training/evaluation/${evaluation}.txt" \
        --output_dir eval_results/${evaluation}/ \
        --override_batch_size 16

    default_location=$(echo $model_to_evaluate | tr "/" "_" )   # this is what lighteval gives
    new_location=$(echo $default_location | rev | cut -f 1-3 -d"_" | rev)  # this results in "IP_global_stepXXXX
    new_save_path=eval_results/${evaluation}/${REGISTER}/${new_location}/
    mkdir -p $new_save_path
    mv eval_results/${evaluation}/results/$default_location/* $new_save_path
done
#rm -r eval_results/${evaluation}/results/$default_location
