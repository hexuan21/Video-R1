echo "Download RL data and videos."
echo "Current RL data name: ${DATASET_NAME}"
python examples/train_full/prepare_data.py \
    --data_name ${DATASET_NAME} \
    --data_save_dir "src/r1-v/data_vs2_grpo" \