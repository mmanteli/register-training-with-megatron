#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=50G
#SBATCH --partition=small-g
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:mi250:1
#SBATCH --account=project_462000615
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000353/amanda/megatron-training/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH


evaluation=$1
REGISTER=$2
STEP=$3

echo $evaluation $REGISTER $STEP

#full path so renaming works
model_to_evaluate="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints_converted/${REGISTER}/iter_${STEP}"


export TRANSFORMERS_CACHE="/scratch/project_462000353/cache"
export HF_HOME="/scratch/project_462000353/cache"

echo "START: $(date)"

case $evaluation in
    "test")
        evaluation="arc_easy_0"
        srun python /scratch/project_462000353/amanda/register-training/pythonuserbase/bin/lighteval accelerate \
            --model_args "pretrained=${model_to_evaluate},tokenizer=gpt2" \
            --tasks "lighteval|arc:easy|0|0" \
            --output_dir eval_results/${evaluation}/ \
            --override_batch_size 16
    ;;
    "fineweb")
        export HF_DATASETS_TRUST_REMOTE_CODE=true
        srun python /scratch/project_462000353/amanda/megatron-training/pythonuserbase/bin/lighteval accelerate \
            --model_args "pretrained=${model_to_evaluate},tokenizer=gpt2,trust_remote_code=True" \
            --custom_tasks "/scratch/project_462000353/amanda/register-training/Lighteval-on-LUMI/evals/tasks/lighteval_tasks.py" \
            --max_samples 1000 \
            --tasks "/scratch/project_462000353/amanda/register-training/Lighteval-on-LUMI/evals/tasks/fineweb.txt" \
            --output_dir eval_results/${evaluation}/ \
            --override_batch_size 16
    ;;
    "leaderboard")
        srun python /scratch/project_462000353/amanda/register-training/pythonuserbase/bin/lighteval accelerate \
            --model_args "pretrained=${model_to_evaluate},tokenizer=gpt2" \
            --tasks "/scratch/project_462000353/amanda/register-training/Lighteval-on-LUMI/examples/tasks/open_llm_leaderboard_tasks.txt" \
            --output_dir eval_results/${evaluation}/ \
            --override_batch_size 16
    ;;
    "preliminary")
        srun python /scratch/project_462000353/amanda/register-training/pythonuserbase/bin/lighteval accelerate \
            --model_args "pretrained=${model_to_evaluate},tokenizer=gpt2" \
            --tasks "/scratch/project_462000353/amanda/register-training/register-model-training/evaluation/multiple.txt" \
            --output_dir eval_results/${evaluation}/ \
            --override_batch_size 16
    ;;
    *)
        echo "invalid evaluation given"
    ;;
esac

echo "END: $(date)"

default_location=$(echo $model_to_evaluate | tr "/" "_" )   # this is what lighteval gives
new_location=$(echo $default_location | rev | cut -f 1-3 -d"_" | rev)  # this results in "IP_iter_XXXXX
new_save_path=eval_results/${evaluation}/${REGISTER}/${new_location}/
mkdir -p $new_save_path
mv eval_results/${evaluation}/results/$default_location/* $new_save_path
#rm -r eval_results/${evaluation}/results/$default_location   
