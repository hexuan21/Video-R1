echo "Download RL data and videos."

python prepare_data.py \
    --json_name grpo_27k \
    --video_zip_name sft_27k_videos \
    --data_save_dir "r1-v/Video-R1-data" 