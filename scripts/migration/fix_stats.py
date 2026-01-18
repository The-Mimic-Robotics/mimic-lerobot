#!/usr/bin/env python3
"""Fix stats.json for all converted datasets - add missing action/observation.state stats."""

import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
import tempfile
import os

# Datasets to fix
STATIONARY_IDS = [1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
MOBILE_IDS = [2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

def compute_stats_from_parquet(repo_id: str) -> dict:
    """Compute action and observation.state stats from parquet files."""
    print(f"  Computing stats from parquet files...")

    # Get list of parquet files
    files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith('.parquet')]

    all_actions = []
    all_states = []

    for pf in parquet_files:
        path = hf_hub_download(repo_id=repo_id, filename=pf, repo_type="dataset")
        table = pq.read_table(path)

        # Get action column
        if "action" in table.column_names:
            actions = table.column("action").to_pylist()
            all_actions.extend(actions)

        # Get state column
        if "observation.state" in table.column_names:
            states = table.column("observation.state").to_pylist()
            all_states.extend(states)

    stats = {}

    if all_actions:
        actions_arr = np.array(all_actions, dtype=np.float32)
        stats["action"] = {
            "mean": actions_arr.mean(axis=0).tolist(),
            "std": actions_arr.std(axis=0).tolist(),
            "min": actions_arr.min(axis=0).tolist(),
            "max": actions_arr.max(axis=0).tolist()
        }
        print(f"    action shape: {actions_arr.shape}")

    if all_states:
        states_arr = np.array(all_states, dtype=np.float32)
        stats["observation.state"] = {
            "mean": states_arr.mean(axis=0).tolist(),
            "std": states_arr.std(axis=0).tolist(),
            "min": states_arr.min(axis=0).tolist(),
            "max": states_arr.max(axis=0).tolist()
        }
        print(f"    observation.state shape: {states_arr.shape}")

    return stats


def fix_dataset_stats(repo_id: str):
    """Fix stats.json for a dataset."""
    print(f"\nFixing {repo_id}...")

    api = HfApi()

    try:
        # Download current stats
        stats_path = hf_hub_download(repo_id=repo_id, filename="meta/stats.json", repo_type="dataset")
        with open(stats_path) as f:
            current_stats = json.load(f)

        # Check if action/state stats already exist with correct shape
        if "action" in current_stats:
            action_len = len(current_stats["action"]["mean"])
            if action_len == 15:
                print(f"  Already has correct 15D stats, skipping")
                return True
            else:
                print(f"  Has {action_len}D stats, will recompute")
        else:
            print(f"  Missing action/state stats")

        # Compute stats from parquet
        new_stats = compute_stats_from_parquet(repo_id)

        # Merge stats
        current_stats.update(new_stats)

        # Upload updated stats
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_file = Path(tmpdir) / "stats.json"
            with open(stats_file, 'w') as f:
                json.dump(current_stats, f, indent=2)

            api.upload_file(
                path_or_fileobj=str(stats_file),
                path_in_repo="meta/stats.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Fix: add action/observation.state stats"
            )

        print(f"  ✓ Fixed!")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("Fixing stats for all datasets...")

    success = 0
    failed = 0

    # Fix stationary datasets
    print("\n=== Stationary Datasets ===")
    for i in STATIONARY_IDS:
        repo_id = f"neryotw/bimanual_blue_block_handover_{i}_v30_15d"
        if fix_dataset_stats(repo_id):
            success += 1
        else:
            failed += 1

    # Fix mobile datasets
    print("\n=== Mobile Datasets ===")
    for i in MOBILE_IDS:
        repo_id = f"neryotw/mobile_bimanual_blue_block_handover_{i}_v30_15d"
        if fix_dataset_stats(repo_id):
            success += 1
        else:
            failed += 1

    print(f"\n=== Summary ===")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
