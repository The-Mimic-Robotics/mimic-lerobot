#!/usr/bin/env python
"""
Batch convert ALL datasets to neryotw/ namespace.
Handles:
1. Stationary (1-26): Converts 12D -> 15D
2. Mobile (1-26): Preserves 15D (just v3.0 update if needed)
"""

import sys
import subprocess
import shutil
from pathlib import Path
from lerobot.datasets.utils import load_info
from lerobot.utils.constants import HF_LEROBOT_HOME
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent / "src"))

DATASETS_TO_PROCESS = []

# 1. Stationary bimanual datasets (1-26)
for i in range(1, 27):
    DATASETS_TO_PROCESS.append({
        "repo_id": f"Mimic-Robotics/bimanual_blue_block_handover_{i}",
        "type": "stationary"
    })

# 2. Mobile bimanual datasets (1-26)
for i in range(1, 27):
    DATASETS_TO_PROCESS.append({
        "repo_id": f"Mimic-Robotics/mobile_bimanual_blue_block_handover_{i}",
        "type": "mobile"
    })

def convert_dataset(repo_id, dataset_type, push_to_hub=True):
    """Convert a single dataset. Returns True on success."""
    
    # NEW name in USER'S namespace
    dataset_name = repo_id.split("/")[-1]
    
    # Validation: Skip "42" aggregate if it exists in list
    if dataset_name == "42": 
        return True

    output_repo_id = f"neryotw/{dataset_name}_v30_15d"
    
    # Check if already exists on Hub
    api = HfApi()
    try:
        if api.repo_exists(repo_id=output_repo_id, repo_type="dataset"):
            print(f"  [Skip] Alrady exists on Hub: {output_repo_id}")
            return True
    except Exception as e:
        print(f"  Warning checking Hub: {e}")

    print(f"\n{'='*60}")
    print(f"Processing: {repo_id} ({dataset_type})")
    print(f"Output:     {output_repo_id}")
    print('='*60)
    
    local_path = HF_LEROBOT_HOME / repo_id
    
    # Clear cache to force fresh download
    if local_path.exists():
        print("  Clearing local cache for fresh download...")
        try:
            shutil.rmtree(local_path)
        except Exception as e:
            print(f"  ⚠ Warning: Could not clear cache: {e}")
    
    env = {
        "PYTHONPATH": f"{Path.cwd()}/src:/Users/Nicholas/Library/Python/3.11/lib/python/site-packages",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(Path.home()),
    }
    
    # Step 1: Convert v2.1 to v3.0
    print("\n[Step 1/2] Converting v2.1 -> v3.0...")
    cmd1 = [
        "python3.11", "-m", "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
        "--push-to-hub=false"
    ]
    
    result = subprocess.run(cmd1, cwd=Path.cwd(), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        if "404 Client Error" in result.stderr:
            print(f"  ⚠ Dataset not found on Hub (skipping): {repo_id}")
            return True 
        
        print(f"  ✗ v2.1->v3.0 failed: {result.stderr[:300]}")
        return False
    print("  ✓ v3.0 conversion complete")
    
    # Step 2: Handle Dimensions
    if dataset_type == "stationary":
        print("\n[Step 2/2] Converting 12D -> 15D (Stationary)...")
        cmd2 = [
            "python3.11", "convert_dataset_dims.py",
            f"--repo-id={repo_id}",
            f"--output-repo-id={output_repo_id}",
        ]
        if push_to_hub:
            cmd2.append("--push")
        
        result = subprocess.run(cmd2, cwd=Path.cwd(), env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
             print(f"  ✗ 12D->15D failed: {result.stderr[:300]}")
             return False
        print("  ✓ 15D conversion complete")
        
    else: # mobile
        print("\n[Step 2/2] Verifying/Pushing 15D (Mobile)...")
        # For mobile, we verify it is 15D and push
        # NOTE: convert_dataset_v21_to_v30 creates a new folder with _v30 suffix
        converted_repo_id = f"{repo_id}_v30"
        
    
    
        push_script = f"""
import sys
from pathlib import Path
sys.path.insert(0, "{Path.cwd()}/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from huggingface_hub import hf_hub_download

repo_id = "{repo_id}"
converted_repo_id = "{converted_repo_id}" # repo_id + '_v30'

# Logic: use converted local copy if valid, else original
local_converted = HF_LEROBOT_HOME / converted_repo_id
if (local_converted / "meta/info.json").exists():
    use_repo = converted_repo_id
else:
    # Fallback to original repo_id. Ensure metadata is downloaded!
    use_repo = repo_id
    print(f"Fallback to original: {{repo_id}}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, allow_patterns=["meta/*"], repo_type="dataset", force_download=True)
    except Exception as e:
        print(f"Warning downloading metadata: {{e}}")

try:
    print(f"Loading dataset: '{{use_repo}}'")
    ds = LeRobotDataset(use_repo) 
    # Verify dimensions
    if ds.meta.features['action']['shape'][0] != 15:
        print(f"ERROR: Expected 15D, got {{ds.meta.features['action']['shape'][0]}}")
        sys.exit(1)
    ds.push_to_hub(repo_id="{output_repo_id}")

except Exception as e:
    print(f"LeRobotDataset load failed: {{e}}")
    print("Attempting manual verification via info.json...")
    import json
    from huggingface_hub import HfApi
    
    # Try to find info.json in the local snapshot
    # HF_LEROBOT_HOME / repo_id might be a symlink dir or contain 'meta'
    info_path = HF_LEROBOT_HOME / use_repo / "meta/info.json"
    if not info_path.exists():
         info_path = HF_LEROBOT_HOME / use_repo / "info.json"
    
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Check dimensions (nested structure: features -> action -> shape)
        try:
            shape = info['features']['action']['shape'][0]
            if shape == 15:
                print("  ✓ Manually Verified 15D")
                # Push manually
                print(f"Pushing to {{'{output_repo_id}'}} via upload_folder...")
                api = HfApi()
                api.create_repo(repo_id="{output_repo_id}", repo_type="dataset", exist_ok=True)
                api.upload_folder(
                    repo_id="{output_repo_id}",
                    folder_path=str(HF_LEROBOT_HOME / use_repo),
                    repo_type="dataset",
                    ignore_patterns=[".git", ".cache"]
                )
                print("  ✓ Pushed successfully")
            else:
                print(f"ERROR: Manual verify found {{shape}}D, expected 15D")
                sys.exit(1)
        except KeyError as ke:
             print(f"Error parsing info.json: {{ke}}")
             sys.exit(1)
    else:
        print(f"Could not find info.json at {{info_path}}")
        sys.exit(1)
"""
        result = subprocess.run(["python3.11", "-c", push_script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
        if result.returncode != 0:
             print(f"  ✗ Mobile push failed: {result.stderr[:300]} {result.stdout[:300]}")
             return False
        print("  ✓ Verified 15D and Pushed")

    if push_to_hub and dataset_type == "stationary":
         print(f"  ✓ Pushed to: {output_repo_id}")
    
    return True

def main():
    print("\n" + "="*60)
    print("BATCH DATASET CONVERSION (FULL)")
    print("="*60)
    print(f"Datasets waiting: {len(DATASETS_TO_PROCESS)}")
    
    successes = []
    failures = []
    
    for item in DATASETS_TO_PROCESS:
        try:
            if convert_dataset(item["repo_id"], item["type"], push_to_hub=True):
                successes.append(item["repo_id"])
            else:
                failures.append(item["repo_id"])
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            failures.append(item["repo_id"])
    
    # Summary
    print("\n" + "="*60)
    print("BATCH CONVERSION SUMMARY")
    print("="*60)
    print(f"✓ Successful: {len(successes)}")
    print(f"✗ Failed:     {len(failures)}")
    for f in failures:
        print(f"    {f}")
    
    return len(failures) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
