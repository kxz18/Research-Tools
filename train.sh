#!/bin/zsh
##########################################################################
# File Name: train.sh
# Author: kxz
# mail: 15068701650@163.com
# Created Time: Saturday, December 11, 2021 PM04:49:45 HKT
#########################################################################

########## adjust configs according to your needs ##########
DATA_DIR=/data/private/kxz/prompt/webnlg_2017
TRAIN_SET=${DATA_DIR}/train
DEV_SET=${DATA_DIR}/dev
SAVE_DIR=${DATA_DIR}/ckpt
BATCH_SIZE=32
######### end of adjust ##########

########## Instruction ##########
# This script takes three optional environment variables:
# GPU / ADDR / PORT
# e.g. Use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########

# set master address and port e.g. ADDR=localhost PORT=9901 bash train.sh
MASTER_ADDR=localhost
MASTER_PORT=9901
if [ $ADDR ]; then MASTER_ADDR=$ADDR; fi
if [ $PORT ]; then MASTER_PORT=$PORT; fi
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"

# set gpu, e.g. GPU="0,1,2,3" bash train.sh
if [ -z "$GPU" ]; then
    GPU="-1"  # use CPU
fi
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPUs: $GPU"
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ ${#GPU_ARR[@]} -gt 1 ]; then
	export OMP_NUM_THREADS=2
	PREFIX="torchrun --nproc_per_node=${#GPU_ARR[@]} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --standalone --nnodes=1"
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
