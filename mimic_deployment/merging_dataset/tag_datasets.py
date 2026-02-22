#!/usr/bin/env python3
"""
Script to tag datasets with required codebase versions for LeRobot compatibility
"""
from huggingface_hub import HfApi
import json
import os
from pathlib import Path

# Datasets that need tagging
DATASETS_TO_TAG = [
    "Mimic-Robotics/test2",
    "Mimic-Robotics/test1",
    "Mimic-Robotics/mimic_tictactoe_blueO_full30hz",
    "Mimic-Robotics/mimic_tictactoe_blue_o_handover_top_right_v2"
]

def get_dataset_version(repo_id: str) -> str:
    """Get the codebase_version from a dataset's info.json file"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    info_path = cache_dir / "meta" / "info.json"
    
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
            if "codebase_version" in info:
                return info["codebase_version"]
    
    # Default to v3.0 if not found (most recent format)
    return "v3.0"

def tag_dataset(repo_id: str, version_tag: str):
    """Tag a dataset with the codebase version"""
    api = HfApi()
    
    try:
        print(f"Tagging {repo_id} with version {version_tag}...", end=" ")
        api.create_tag(
            repo_id=repo_id,
            tag=version_tag,
            repo_type="dataset",
            exist_ok=True,  # Don't fail if tag already exists
        )
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 70)
    print("LeRobot Dataset Tagging Script")
    print("=" * 70)
    print(f"\nFound {len(DATASETS_TO_TAG)} datasets to tag\n")
    
    success = 0
    failed = 0
    
    for repo_id in DATASETS_TO_TAG:
        version = get_dataset_version(repo_id)
        if tag_dataset(repo_id, version):
            success += 1
        else:
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Tagging Summary: {success} succeeded, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All datasets tagged successfully!")
        print("You can now run training with: ./train_manager.sh --policy act --dataset-group red_x_handover_and_place_center")
    else:
        print(f"\n✗ {failed} datasets failed to tag. Please check your Hugging Face credentials and permissions.")

if __name__ == "__main__":
    main()
