echo "Download RL data and videos."

python prepare_data.py \
    --json_name grpo_25k \
    --video_zip_name sft_25k_videos \
    --data_save_dir "r1-v/Video-R1-data" 