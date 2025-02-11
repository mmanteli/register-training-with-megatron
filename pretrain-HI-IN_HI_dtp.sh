#!/bin/bash

#SBATCH --job-name=HI-IN_HI_dtp-1.71B
#SBATCH --nodes=16
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=01-23:57:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# HI-IN HI dtp
REGISTER_1=HI-IN
REGISTER_2=HI
REGISTER_3=dtp
REGISTER="${REGISTER_1}_${REGISTER_2}_${REGISTER_3}"
MEGATRON_PATH="/scratch/project_462000353/amanda/megatron-training/Megatron-LM-lumi"

mkdir -p workdir
wd=$(realpath workdir)

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

# log starts
# ./log_restart_info.sh | tee -a starts.log

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup
CONTAINER="/scratch/project_462000353/containers/flashattention_v2_new"
SING_BIND="/scratch/project_462000353,/flash/project_462000353"


LEARNING_RATE=3e-4

set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" logs/latest_${REGISTER}.out
ln -f -s "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" logs/latest_${REGISTER}.err


CHECKPOINT_PATH="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints/${REGISTER}"
TENSORBOARD_PATH="tensorboard/${REGISTER}.$SLURM_JOB_ID"
#rm -rf "$CHECKPOINT_PATH" "$TENSORBOARD_PATH" # Start from scratch

export CUDA_DEVICE_MAX_CONNECTIONS=1

#TRAIN_DATA_PATH="/scratch/project_462000353/HPLT-REGISTERS/samples-150B-by-register-xlmrl/tokenized"
TRAIN_DATA_PATH="/flash/project_462000353/registers"
data1="0.34 ${TRAIN_DATA_PATH}/${REGISTER_1}/eng_Latn_text_document"
data2="0.33 ${TRAIN_DATA_PATH}/${REGISTER_2}/eng_Latn_text_document"
data3="0.33 ${TRAIN_DATA_PATH}/${REGISTER_3}/eng_Latn_text_document"
#TRAIN_DATA="0.2 ${TRAIN_DATA_PATH}/${REGISTER_1}/eng_Latn_text_document 0.5 ${TRAIN_DATA_PATH}/${REGISTER_2}/eng_Latn_text_document"
TRAIN_DATA="${data1} ${data2} ${data3}"
# YOU CAN DEFINE SAMPLING like this:
#TRAIN_DATA='0.5 dataset1, 0.5 dataset2'
# and validation like this:
#VALIDATION_DATA=""

# TOKENIZER given as merges and vocab
MERGES=/scratch/project_462000353/tokenizers/gpt2/merges.txt
VOCAB=/scratch/project_462000353/tokenizers/gpt2/vocab.json

PP_SIZE=1
TP_SIZE=1

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=1024

NLAYERS=24
NHIDDEN=2048
NHEADS=32
FFN_HIDDEN_SIZE=8192
SEQ_LEN=2048

export MEMORY_OPT_ALLREDUCE_SIZE=150000000
echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

# TOTAL_TOKENS=2_000_000_000_000 # 2 trillion
TOTAL_TOKENS=350_000_000_000 # 350B
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((GLOBAL_BATCH_SIZE*500))

LOG_INTERVAL=100
SAVE_INTERVAL=1000
EVAL_INTERVAL=4000   # eval does not work with validation data undefined
EVAL_STEPS=100

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr $LEARNING_RATE \
    --min-lr 3e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --bf16 \
    --disable-bias-linear \
    --init-method-std 0.0048 \
    --make-vocab-size-divisible-by 128 \
    --no-gradient-accumulation-fusion \
    --normalization RMSNorm \
    --seed 42 \
    --swiglu \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-query-key-layer-scaling \
    --no-masked-softmax-fusion \
    --sequence-parallel \
    $OPTIMIZER_ARGS \
    "
#--use-distributed-optimizer \
#--untie-embeddings-and-output-weights \  # this is the discrepancy between 1.71B and 1.82B between Fineweb paper versions

OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "
#    --wandb-name v3-7B \

CMD=" \
    ${MEGATRON_PATH}/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --train-data-path $TRAIN_DATA \
    --dataloader-type single \
    --num-workers 0 \
    "
    # --data-impl mmap \
    # --valid-data-path $VALIDATION_DATA \

c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Bind mask for two threads per core
#BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

# add a pythonuserbase to an empty dir to avoid problems with user's local
# python install being imported into the singularity container.
#mkdir -p pythonuserbase
export PYTHONUSERBASE=/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/pythonuserbase #/scratch/project_462000353/avirtanen/Megatron-LM-lumi-head/pythonuserbase/lib/python3.10/site-packages/

echo $CMD

echo "START $SLURM_JOBID: $(date)"

if [ ! -d "$wd"/cray-deps ] ; then
  rm -rf "$wd"/cray-deps
  mkdir "$wd"/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    -B /opt/cray:/opt/cray \
    -B "$wd"/cray-deps:/opt/cray-deps \
    -B "$wd":/workdir \
    -B "$SING_BIND" \
    "$CONTAINER" \
    ${MEGATRON_PATH}/launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"
cp "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "logs/done/${REGISTER}-1.71B-${SLURM_JOB_ID}.out"
cp "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" "logs/done/${REGISTER}-1.71B-${SLURM_JOB_ID}.err"
exit 0