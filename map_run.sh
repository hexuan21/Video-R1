#!/bin/bash

source /map-vepfs/miniconda3/bin/activate
conda activate video_r1_xuan

export RANK=$MLP_ROLE_INDEX
export WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_WORKER_GPU))
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export LOCAL_RANK=$(($RANK % $MLP_WORKER_GPU))

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

unset https_proxy; unset http_proxy
export https_proxy="http://100.64.117.161:3128"
export http_proxy="http://100.64.117.161:3128"

cd /map-vepfs/xuan/Video-R1

export HF_HOME='/map-vepfs/huggingface'
bash /map-vepfs/xuan/Video-R1/src/scripts/run_grpo_video.sh
