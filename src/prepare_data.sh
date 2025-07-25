echo "Download RL data and videos."

python prepare_data.py \
    --json_name rl_17k \
    --video_zip_name sft_17k_videos \
    --data_save_dir "r1-v/data_vs2_grpo" 