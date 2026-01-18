#!/usr/bin/env python
"""
Batch convert all legacy bimanual datasets from v2.1/12D to v3.0/15D.
Creates NEW datasets with suffix, never overwrites originals.
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Legacy datasets to convert (v2.1, 12D)
LEGACY_DATASETS = [
    "Mimic-Robotics/bimanual_blue_block_handover_1",
    "Mimic-Robotics/bimanual_blue_block_handover_2",
    "Mimic-Robotics/bimanual_blue_block_handover_3",
    "Mimic-Robotics/bimanual_blue_block_handover_4",
    "Mimic-Robotics/bimanual_blue_block_handover_5",
    "Mimic-Robotics/bimanual_blue_block_handover_6",
    "Mimic-Robotics/bimanual_blue_block_handover_7",
    "Mimic-Robotics/bimanual_blue_block_handover_8",
    "Mimic-Robotics/bimanual_blue_block_handover_9",
    "Mimic-Robotics/bimanual_blue_block_handover_10",
    "Mimic-Robotics/bimanual_blue_block_handover_11",
    "Mimic-Robotics/bimanual_blue_block_handover_12",
]

def convert_dataset(repo_id, push_to_hub=True):
    """Convert a single dataset. Returns True on success."""
    
    # NEW name in USER'S namespace - never overwrites originals
    dataset_name = repo_id.split("/")[-1]
    output_repo_id = f"neryotw/{dataset_name}_v30_15d"
    
    print(f"\n{'='*60}")
    print(f"Converting: {repo_id}")
    print(f"Output:     {output_repo_id}")
    print('='*60)
    
    from lerobot.datasets.utils import load_info
    from lerobot.utils.constants import HF_LEROBOT_HOME
    import shutil
    
    local_path = HF_LEROBOT_HOME / repo_id
    
    # Clear cache to force fresh download (avoids incomplete downloads)
    if local_path.exists():
        print("  Clearing local cache for fresh download...")
        shutil.rmtree(local_path)
    
    # Always need v3.0 conversion after clearing cache
    needs_v30_conversion = True
    
    env = {
        "PYTHONPATH": f"{Path.cwd()}/src:/Users/Nicholas/Library/Python/3.11/lib/python/site-packages",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(Path.home()),
    }
    
    # Step 1: Convert v2.1 to v3.0 (if needed)
    if needs_v30_conversion:
        print("\n[Step 1/2] Converting v2.1 -> v3.0...")
        cmd1 = [
            "python3.11", "-m", "lerobot.datasets.v30.convert_dataset_v21_to_v30",
            f"--repo-id={repo_id}",
            "--push-to-hub=false"  # Don't push intermediate
        ]
        
        result = subprocess.run(cmd1, cwd=Path.cwd(), env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ✗ v2.1->v3.0 failed: {result.stderr[:300]}")
            return False
        print("  ✓ v3.0 conversion complete")
    else:
        print("\n[Step 1/2] Skipped (already v3.0)")
    
    # Step 2: Convert 12D to 15D
    print("\n[Step 2/2] Converting 12D -> 15D...")
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
        print(f"  stdout: {result.stdout[-300:]}")
        return False
    print("  ✓ 15D conversion complete")
    
    if push_to_hub:
        print(f"  ✓ Pushed to: {output_repo_id}")
    
    return True

def main():
    print("\n" + "="*60)
    print("BATCH DATASET CONVERSION")
    print("="*60)
    print(f"Datasets to convert: {len(LEGACY_DATASETS)}")
    print("Output naming: <original>_v30_15d (no overwrites)")
    
    successes = []
    failures = []
    
    for repo_id in LEGACY_DATASETS:
        try:
            if convert_dataset(repo_id, push_to_hub=True):
                successes.append(repo_id)
            else:
                failures.append(repo_id)
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            failures.append(repo_id)
    
    # Summary
    print("\n" + "="*60)
    print("BATCH CONVERSION SUMMARY")
    print("="*60)
    print(f"✓ Successful: {len(successes)}")
    for s in successes:
        print(f"    {s}")
    print(f"✗ Failed: {len(failures)}")
    for f in failures:
        print(f"    {f}")
    
    return len(failures) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
