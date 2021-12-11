#!/bin/zsh
##########################################################################
# File Name: train.sh
# Author: kxz
# mail: 15068701650@163.com
# Created Time: Saturday, December 11, 2021 PM04:49:45 HKT
#########################################################################
DATA_DIR=/data/private/kxz/prompt/webnlg_2017
########## adjust configs according to your needs ##########
TRAIN_SET=${DATA_DIR}/train
DEV_SET=${DATA_DIR}/dev
SAVE_DIR=${DATA_DIR}/ckpt
BATCH_SIZE=32


# set gpu, e.g. GPU="0,1,2,3" bash train.sh
if [ -z "$GPU" ]; then
    GPU="-1"  # use CPU
fi
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPUs: $GPU"
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ ${#GPU_ARR[@]} -gt 1 ]; then
    PREFIX="python -m torch.distributed.launch --nproc_per_node=${#GPU_ARR[@]}"
else
    PREFIX="python"
fi

${PREFIX} train.py \
    --train_set $TRAIN_SET \
    --valid_set $DEV_SET \
    --save_dir $SAVE_DIR \
    --shuffle \
    --batch_size ${BATCH_SIZE} \
    --gpu "${!GPU_ARR[@]}"