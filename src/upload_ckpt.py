from huggingface_hub import HfApi, upload_file
import os
import shutil
base_model_or_dir = "/home/user/.cache/huggingface/hub/models--videoscore2--vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192/snapshots/ab4add7aa8f3c4da96f017b06e3d201be9302264"
checkpoint_dir = "/data/xuan/workdir/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO/checkpoint-800"
repo_id = f"videoscore2/vs2_qwen2_5vl_grpo_17k_try_1e-6_800"
upload_dir = "./model_upload_dir"
token=os.environ["HF_TOKEN"]
repo_type="model"

if os.path.exists(upload_dir):
    shutil.rmtree(upload_dir)

shutil.copytree(base_model_or_dir, upload_dir)
shutil.copy(f"{checkpoint_dir}/model-00001-of-00004.safetensors", upload_dir)
shutil.copy(f"{checkpoint_dir}/model-00002-of-00004.safetensors", upload_dir)
shutil.copy(f"{checkpoint_dir}/model-00003-of-00004.safetensors", upload_dir)
shutil.copy(f"{checkpoint_dir}/model-00004-of-00004.safetensors", upload_dir)
shutil.copy(f"{checkpoint_dir}/model.safetensors.index.json", upload_dir)


api = HfApi(token=token)
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Repository '{repo_id}' already exists.")
except Exception as e:
    print(f"🚧 Repository '{repo_id}' not found. Creating a new public repo...")
    api.create_repo(
        repo_id=repo_id,
        token=token,
        repo_type=repo_type,
        private=False
    )

for filename in os.listdir(upload_dir):
    local_path = os.path.join(upload_dir, filename)
    if os.path.isfile(local_path):
        print(f"Uploading {filename}...")
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,         
            repo_id=repo_id,
            repo_type="model"
        )