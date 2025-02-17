#!/bin/bash
#SBATCH --job-name=baseline_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=50G
#SBATCH --partition=small-g
#SBATCH --time=0:45:00
#SBATCH --gres=gpu:mi250:1
#SBATCH --account=project_462000353
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000353/amanda/megatron-training/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH


evaluation=$1
model=$2
STEP=$3

case $model in
    "fineweb")
        model_to_evaluate="/scratch/project_462000353/pyysalos/second-hplt-eval/fineweb_iter_${STEP}"
        if ! [ -d $model_to_evaluate ]; then
            echo "Model path does not exist."
            exit 1
        fi
    ;;
    "hplt-v2-dedup")
        model_to_evaluate="/scratch/project_462000353/pyysalos/second-hplt-eval/hplt_v2_dedup_iter_${STEP}"
        if ! [ -d $model_to_evaluate ]; then
            echo "Model path does not exist."
            exit 1
        fi
    ;;
    *)
        echo "error in finding model"
        exit 1
    ;;
esac

sleep_time="${4:-0}"
sleep $sleep_time   # this is because there is a problem with the jobs (+200) starting simulateneously and crashing

echo $evaluation $model $STEP

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



# default location looks like this: _scratch_project_462000353_pyysalos_second-hplt-eval_fineweb_iter_0001000
default_location=$(echo $model_to_evaluate | tr "/" "_" )   # this is what lighteval gives
new_location=$(echo $default_location | rev | cut -f 1-3 -d"_" | rev)  # this results in "IP_iter_XXXXX
new_save_path=eval_results/${evaluation}/${model}/${new_location}/
mkdir -p $new_save_path
mv eval_results/${evaluation}/results/$default_location/* $new_save_path
#rm -r eval_results/${evaluation}/results/$default_location 

# move logs
mv logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out logs/done/${evaluation}/${model}-${STEP}-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
mv logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err logs/done/${evaluation}/${model}-${STEP}-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err
