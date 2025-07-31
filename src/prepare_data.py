import os
import shutil
import json
import argparse
import requests
from tqdm import tqdm
from zipfile import ZipFile, BadZipFile

HF_DATASET_USER = "hexuan21"
HF_DATASET_NAME = "vs2_rl_data"
HF_VIDEO_NAME   = "vs2_sft_video"
CHUNK = 1 << 14  # 16 KB


def download_file(url: str, save_path: str, overwrite: bool = False, timeout: int = 15):
    if os.path.exists(save_path) and not overwrite:
        print(f"[skip] {save_path} already exists")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(save_path))
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()
    print(f"[ok] Downloaded → {save_path}")



def main(json_name: str, video_zip_name:str, data_save_dir: str):
    data_file  = f"{json_name}.json"
    data_save = os.path.join(data_save_dir, data_file)
    
    video_zip  = f"{video_zip_name}.zip"
    video_dir = os.path.join(data_save_dir, "vs2_videos")
    zip_save = os.path.join(data_save_dir, video_zip)
    
    data_url = f"https://huggingface.co/datasets/{HF_DATASET_USER}/{HF_DATASET_NAME}/resolve/main/{data_file}"
    video_zip_url = f"https://huggingface.co/datasets/{HF_DATASET_USER}/{HF_VIDEO_NAME}/resolve/main/{video_zip}"
    
    download_file(data_url, data_save, overwrite=True)
    with open(data_save, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(data_save, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    download_file(video_zip_url, zip_save, overwrite=True)

    os.makedirs(video_dir, exist_ok=True)
    try:
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        with ZipFile(zip_save) as zf:
            zf.extractall(video_dir)
        print(f"[ok] Unzipped → {video_dir}")
    except BadZipFile as e:
        print(f"[error] Bad zip file: {e}")
    os.remove(zip_save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RL JSON & ZIP")
    parser.add_argument("--json_name", required=True)
    parser.add_argument("--video_zip_name", required=True)
    parser.add_argument("--data_save_dir", required=True)
    args = parser.parse_args()
    main(args.json_name,args.video_zip_name,args.data_save_dir)