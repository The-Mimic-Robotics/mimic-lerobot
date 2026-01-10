#!/usr/bin/env python3
"""
Convert old bimanual datasets (12D arms) to new mobile bimanual format (15D arms+base).
Zero-pads base velocities (vx, vy, omega) for stationary tasks.

Usage:
    python convert_old_to_mobile_dataset.py --old-repo "Mimic-Robotics/bimanual_blue_block_handover_1" \
                                             --new-repo "Mimic-Robotics/mobile_bimanual_blue_block_handover_1" \
                                             [--test-only]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def convert_dataset(old_repo_id: str, new_repo_id: str, test_only: bool = False):
    """Convert old 12D dataset to new 15D format by zero-padding base velocities."""
    
    print(f"\n{'='*70}")
    print(f"Converting: {old_repo_id}")
    print(f"       To: {new_repo_id}")
    print(f"{'='*70}\n")
    
    # Download old dataset
    print("Downloading old dataset...")
    old_cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / old_repo_id
    snapshot_download(
        repo_id=old_repo_id,
        repo_type="dataset",
        local_dir=old_cache_dir,
    )
    
    # Check if dataset needs format conversion from v2.1 to v3.0
    info_path = old_cache_dir / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    dataset_version = info.get("codebase_version", "v2.1")
    if dataset_version.startswith("v2"):
        print(f"WARNING: Dataset is in {dataset_version} format, converting to v3.0...")
        import subprocess
        
        # Use CLI to convert
        result = subprocess.run(
            [
                "python", "-m", "lerobot.datasets.v30.convert_dataset_v21_to_v30",
                f"--repo-id={old_repo_id}",
                "--push-to-hub=false",
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR: Format conversion failed:")
            print(result.stderr)
            raise RuntimeError("Format conversion failed")
        
        print("Format conversion complete")
    
    # Load old dataset
    print("Loading old dataset...")
    old_dataset = LeRobotDataset(old_repo_id)
    
    # Verify old action dimension
    old_action_dim = old_dataset.meta.info["features"]["action"]["shape"][0]
    print(f"Old action dimension: {old_action_dim}")
    
    if old_action_dim != 12:
        raise ValueError(f"Expected 12D actions, got {old_action_dim}D. Wrong dataset?")
    
    # Create new dataset directory
    new_cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / new_repo_id
    
    if new_cache_dir.exists():
        print(f"WARNING: New dataset already exists at {new_cache_dir}")
        if not test_only:
            response = input("Delete and recreate? (yes/no): ")
            if response.lower() != "yes":
                print("Aborted.")
                return
        shutil.rmtree(new_cache_dir)
    
    print(f"Creating new dataset directory: {new_cache_dir}")
    new_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy everything except data/
    print("Copying metadata and videos...")
    for item in old_cache_dir.iterdir():
        if item.name != "data":
            if item.is_dir():
                shutil.copytree(item, new_cache_dir / item.name)
            else:
                shutil.copy2(item, new_cache_dir / item.name)
    
    # Update info.json with new action dimension
    print("Updating info.json...")
    info_path = new_cache_dir / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Update action feature
    info["features"]["action"]["shape"] = [15]
    info["features"]["action"]["names"] = [
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos",
        "base_vx",
        "base_vy",
        "base_omega"
    ]
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("Updated action dimension: 12 -> 15")
    print("Added base velocity names: base_vx, base_vy, base_omega")
    
    # Convert data files
    print("\nConverting data files (zero-padding actions)...")
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    old_data_dir = old_cache_dir / "data"
    new_data_dir = new_cache_dir / "data"
    new_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each parquet file
    parquet_files = list(old_data_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to convert")
    
    for i, old_file in enumerate(parquet_files, 1):
        # Read old parquet
        table = pq.read_table(old_file)
        data = table.to_pydict()
        
        # Convert actions: add 3 zeros for base velocities
        old_actions = np.array(data["action"])  # Shape: (N, 12)
        base_zeros = np.zeros((old_actions.shape[0], 3), dtype=np.float32)
        new_actions = np.concatenate([old_actions, base_zeros], axis=1)  # Shape: (N, 15)
        
        # Update data dict
        data["action"] = new_actions.tolist()
        
        # Write new parquet
        new_file = new_data_dir / old_file.relative_to(old_data_dir)
        new_file.parent.mkdir(parents=True, exist_ok=True)
        
        new_table = pa.Table.from_pydict(data)
        pq.write_table(new_table, new_file, compression='snappy')
        
        print(f"  [{i}/{len(parquet_files)}] {old_file.name}")
    
    # Update stats.json if it exists
    stats_path = new_cache_dir / "meta" / "stats.json"
    if stats_path.exists():
        print("\nUpdating stats.json...")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Add zero stats for base velocities
        if "action" in stats:
            for stat_type in ["min", "max", "mean", "std"]:
                if stat_type in stats["action"]:
                    old_stat = stats["action"][stat_type]
                    if stat_type in ["min", "max"]:
                        zero_stat = [0.0, 0.0, 0.0]  # Base velocities are zero
                    else:  # mean, std
                        zero_stat = [0.0, 0.0, 0.0]
                    stats["action"][stat_type] = old_stat + zero_stat
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("Extended stats to 15 dimensions")
    
    # Verify conversion from local files
    print("\nVerifying converted dataset...")
    from lerobot.datasets.utils import load_info, load_stats
    import pyarrow.parquet as pq
    
    # Verify info.json
    new_info = load_info(new_cache_dir)
    new_action_dim = new_info["features"]["action"]["shape"][0]
    print(f"New action dimension: {new_action_dim}")
    assert new_action_dim == 15, f"Expected 15, got {new_action_dim}"
    
    # Verify stats.json
    new_stats = load_stats(new_cache_dir)
    print(f"Stats dimensions: mean={new_stats['action']['mean'].shape[0]}, std={new_stats['action']['std'].shape[0]}")
    
    # Sample from parquet to verify zero-padding
    parquet_files = sorted(new_cache_dir.glob("data/**/*.parquet"))
    if parquet_files:
        table = pq.read_table(parquet_files[0])
        actions = table["action"].to_numpy()
        sample_action = actions[0]
        print(f"Sample action shape: {sample_action.shape}")
        print(f"Base velocities (last 3): {sample_action[-3:].tolist()}")
        
        if not np.allclose(sample_action[-3:], np.zeros(3)):
            print("WARNING: Base velocities are not zero!")
        else:
            print("Base velocities confirmed as zeros")
    
    # Summary
    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Episodes: {old_dataset.num_episodes}")
    print(f"Frames: {old_dataset.num_frames}")
    print(f"Old actions: 12D -> New actions: 15D")
    print(f"Dataset saved to: {new_cache_dir}")
    
    if not test_only:
        print("\nNext step: Push to HuggingFace Hub")
        print(f"   huggingface-cli upload {new_repo_id} {new_cache_dir} --repo-type=dataset")
    
    return new_cache_dir


def main():
    parser = argparse.ArgumentParser(description="Convert old bimanual dataset to mobile format")
    parser.add_argument("--old-repo", required=True, help="Old dataset repo ID (e.g., Mimic-Robotics/bimanual_blue_block_handover_1)")
    parser.add_argument("--new-repo", required=True, help="New dataset repo ID (e.g., Mimic-Robotics/mobile_bimanual_blue_block_handover_1)")
    parser.add_argument("--test-only", action="store_true", help="Test conversion without prompting")
    
    args = parser.parse_args()
    
    try:
        convert_dataset(args.old_repo, args.new_repo, args.test_only)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
