#!/usr/bin/env python3
"""Resize old datasets from 480x640 to 720x1280 to match new datasets."""

import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
import json
import shutil

TARGET_HEIGHT = 720
TARGET_WIDTH = 1280

OLD_DATASETS = [
    'neryotw/bimanual_blue_block_handover_1_drift',
    'neryotw/bimanual_blue_block_handover_2_drift',
    'neryotw/bimanual_blue_block_handover_3_drift',
    'neryotw/bimanual_blue_block_handover_4_drift',
    'neryotw/bimanual_blue_block_handover_5_drift',
    'neryotw/bimanual_blue_block_handover_6_drift',
    'neryotw/bimanual_blue_block_handover_7_drift',
    'neryotw/bimanual_blue_block_handover_14_drift',
    'neryotw/bimanual_blue_block_handover_15_drift',
    'neryotw/bimanual_blue_block_handover_16_drift',
    'neryotw/bimanual_blue_block_handover_17_drift',
    'neryotw/bimanual_blue_block_handover_18_drift',
    'neryotw/bimanual_blue_block_handover_19_drift',
    'neryotw/bimanual_blue_block_handover_20_drift',
    'neryotw/bimanual_blue_block_handover_21_drift',
    'neryotw/bimanual_blue_block_handover_22_drift',
    'neryotw/bimanual_blue_block_handover_23_drift',
    'neryotw/bimanual_blue_block_handover_24_drift',
    'neryotw/bimanual_blue_block_handover_25_drift',
    'neryotw/bimanual_blue_block_handover_26_drift',
    'neryotw/mimic_mobile_bimanual_drift_v2_drift',
]

def resize_video(input_path, output_path, width, height):
    """Resize video using ffmpeg."""
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def process_dataset(repo_id):
    """Download, resize videos, and prepare for upload."""
    print(f'\nProcessing {repo_id}...')

    # Download dataset
    local_dir = Path(f'/tmp/resize_datasets/{repo_id.split("/")[1]}')
    if local_dir.exists():
        shutil.rmtree(local_dir)

    snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        local_dir=local_dir,
    )

    # Find and resize all video files
    videos_dir = local_dir / 'videos'
    if videos_dir.exists():
        for video_file in videos_dir.rglob('*.mp4'):
            print(f'  Resizing {video_file.name}...')
            temp_file = video_file.with_suffix('.temp.mp4')
            resize_video(video_file, temp_file, TARGET_WIDTH, TARGET_HEIGHT)
            video_file.unlink()
            temp_file.rename(video_file)

    # Update info.json with new resolution
    info_path = local_dir / 'meta' / 'info.json'
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

        # Update image dimensions in features
        for key, feat in info.get('features', {}).items():
            if 'images' in key and 'shape' in feat:
                # shape is [C, H, W]
                if len(feat['shape']) == 3:
                    feat['shape'] = [feat['shape'][0], TARGET_HEIGHT, TARGET_WIDTH]

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

    # Upload to new repo
    new_repo_id = repo_id + '_resized'
    api = HfApi()

    try:
        api.create_repo(new_repo_id, repo_type='dataset', exist_ok=True)
    except Exception as e:
        print(f'  Repo creation note: {e}')

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=new_repo_id,
        repo_type='dataset',
    )

    print(f'  Uploaded to {new_repo_id}')

    # Cleanup
    shutil.rmtree(local_dir)

if __name__ == '__main__':
    for repo_id in OLD_DATASETS:
        try:
            process_dataset(repo_id)
        except Exception as e:
            print(f'Error processing {repo_id}: {e}')
            continue
