#!/usr/bin/env python3
"""
Batch convert/copy mobile bimanual datasets to neryotw namespace.

These datasets are already 15D native and v3.0 format, so we just need to:
1. Download from Mimic-Robotics
2. Verify 15D action space
3. Push to neryotw namespace with _v30_15d suffix

Usage:
    PYTHONPATH=src python scripts/migration/batch_convert_mobile.py
"""

import sys
import json
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from huggingface_hub import HfApi, snapshot_download
from lerobot.utils.constants import HF_LEROBOT_HOME

# Mobile datasets to convert (IDs 2-7, 14-26 - ID 1 already converted)
MOBILE_IDS = [2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

# Build dataset list
DATASETS_TO_PROCESS = []

# Mobile bimanual datasets
for i in MOBILE_IDS:
    DATASETS_TO_PROCESS.append({
        "repo_id": f"Mimic-Robotics/mobile_bimanual_blue_block_handover_{i}",
        "output_name": f"mobile_bimanual_blue_block_handover_{i}_v30_15d"
    })

# Drift dataset
DATASETS_TO_PROCESS.append({
    "repo_id": "Mimic-Robotics/mimic_mobile_bimanual_drift_v2",
    "output_name": "mimic_mobile_bimanual_drift_v2_v30_15d"
})


def convert_mobile_dataset(repo_id: str, output_name: str) -> bool:
    """
    Copy a mobile dataset to neryotw namespace.
    Mobile datasets are already 15D and v3.0, just need to push to new namespace.

    Returns True on success, False on failure, None if skipped.
    """
    output_repo_id = f"neryotw/{output_name}"
    api = HfApi()

    # Check if already exists on Hub
    try:
        if api.repo_exists(repo_id=output_repo_id, repo_type="dataset"):
            print(f"  [Skip] Already exists: {output_repo_id}")
            return True
    except Exception as e:
        print(f"  Warning checking Hub: {e}")

    # Check if source exists
    try:
        if not api.repo_exists(repo_id=repo_id, repo_type="dataset"):
            print(f"  [Skip] Source not found: {repo_id}")
            return True  # Not a failure, just doesn't exist
    except Exception as e:
        print(f"  Warning checking source: {e}")

    print(f"\n{'='*60}")
    print(f"Processing: {repo_id}")
    print(f"Output:     {output_repo_id}")
    print('='*60)

    # Step 1: Download dataset
    print("\n[Step 1/3] Downloading dataset...")
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=HF_LEROBOT_HOME / repo_id,
            ignore_patterns=["*.lock", ".git*"]
        )
        local_path = Path(local_path)
        print(f"  Downloaded to: {local_path}")
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False

    # Step 2: Verify it's 15D and v3.0
    print("\n[Step 2/3] Verifying dataset...")
    info_path = local_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"  Error: info.json not found at {info_path}")
        return False

    with open(info_path, 'r') as f:
        info = json.load(f)

    # Check version
    version = info.get('codebase_version', 'unknown')
    if version != 'v3.0':
        print(f"  Warning: Expected v3.0, got {version}")
        # Continue anyway for now

    # Check action dimensions
    action_shape = info.get('features', {}).get('action', {}).get('shape', [0])
    if isinstance(action_shape, list):
        action_dims = action_shape[0]
    else:
        action_dims = action_shape

    if action_dims != 15:
        print(f"  Error: Expected 15D actions, got {action_dims}D")
        return False

    print(f"  Version: {version}")
    print(f"  Actions: {action_dims}D")

    # Check cameras
    cameras = [k for k, v in info.get('features', {}).items()
               if v.get('dtype') in ['video', 'image']]
    print(f"  Cameras: {', '.join(cameras)}")

    # Step 3: Push to neryotw namespace
    print("\n[Step 3/3] Pushing to Hub...")
    try:
        api.create_repo(repo_id=output_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=output_repo_id,
            folder_path=str(local_path),
            repo_type="dataset",
            ignore_patterns=[".git", ".cache", "__pycache__", "*.lock"]
        )
        print(f"  Pushed to: {output_repo_id}")
        return True
    except Exception as e:
        print(f"  Error pushing: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("MOBILE DATASET BATCH CONVERSION")
    print("="*60)
    print(f"Datasets to process: {len(DATASETS_TO_PROCESS)}")
    print("  - 19 mobile bimanual (IDs 2-7, 14-26)")
    print("  - 1 drift dataset")
    print()
    print("Note: These datasets are already v3.0 and 15D native.")
    print("      We just need to copy them to neryotw namespace.")
    print()

    successes = []
    failures = []

    for item in DATASETS_TO_PROCESS:
        try:
            result = convert_mobile_dataset(item["repo_id"], item["output_name"])
            if result:
                successes.append(item["repo_id"])
            else:
                failures.append(item["repo_id"])
        except Exception as e:
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
            failures.append(item["repo_id"])

    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Successful: {len(successes)}")
    print(f"Failed:     {len(failures)}")

    if failures:
        print("\nFailed datasets:")
        for f in failures:
            print(f"  - {f}")

    return len(failures) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
