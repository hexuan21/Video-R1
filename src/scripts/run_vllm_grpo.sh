#!/bin/bash


cd src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./log_grpo_17k_base768-768_reward3_temporal_vllm.txt"

DS_CONFIG="local_scripts/zero3.json" 

SFT_Model_Path=videoscore2/vs2_qwen2_5vl_sft_17k_2e-4_2fps_768_768_8192
DATASET_NAME=./Video-R1-data/grpo_17k.json

RUN_NAME=vs2_qwen2_5vl_grpo_17k_1e-6_base768-768_reward3_temporal_vllm
OUTPUT_DIR="./log/$RUN_NAME"

if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
 

# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.
# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="2,4,5,6,7" torchrun \
    --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${SFT_Model_Path} \
    --dataset_name ${DATASET_NAME} \
    --max_prompt_length 16384 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 5 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --save_only_model false \
    --temporal false \
    --len_control true \
    --report_to wandb \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 8 \
    --vllm_device "cuda:0" \
    --vllm_gpu_memory_utilization 0.7 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
