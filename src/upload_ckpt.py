import os
import argparse
import shutil
from huggingface_hub import HfApi, upload_file
from huggingface_hub.utils import RepositoryNotFoundError

def upload_model_with_checkpoint(checkpoint_dir: str, repo_id: str, base_dir: str, token: str):
    # Paths
    base_model_or_dir = os.path.expanduser(
        base_dir,
    )
    upload_dir = "src/model_upload_dir"

    # Clean upload dir
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    shutil.copytree(base_model_or_dir, upload_dir)

    # Copy checkpoint shards
    shards = [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json"
    ]
    for shard in shards:
        src = os.path.join(checkpoint_dir, shard)
        dst = os.path.join(upload_dir, shard)
        shutil.copy(src, dst)

    # Create repo if not exists
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✅ Repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"🚧 Repository '{repo_id}' not found. Creating a new public repo...")
        api.create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            private=False
        )

    # Upload all files
    for filename in os.listdir(upload_dir):
        local_path = os.path.join(upload_dir, filename)
        if os.path.isfile(local_path):
            print(f"📤 Uploading {filename} ...")
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=f"Upload {filename}"
            )

    print(f"✅ Upload completed to {repo_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload safetensors shards + base model to Hugging Face Hub.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo ID")
    parser.add_argument("--base_dir", type=str, required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("❌ HF_TOKEN environment variable not found. Please set it before running this script.")

    upload_model_with_checkpoint(args.checkpoint_dir, args.repo_id, args.base_dir, token)
