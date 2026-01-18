#!/usr/bin/env python3
"""
Train ACT policy on 39 bimanual+mobile datasets with 15D action space.

This trains on:
- 20 stationary bimanual datasets (12D padded to 15D)
- 19 mobile bimanual datasets (native 15D)

Total: 779,133 frames

Usage:
    cd ~/mimic-lerobot
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate lerobot
    python scripts/train_act_15d.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Dataset configuration
STATIONARY_IDS = [1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
MOBILE_IDS = [2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

def build_repo_ids():
    """Build list of all 39 dataset repo IDs."""
    repo_ids = []

    # 20 stationary datasets
    for i in STATIONARY_IDS:
        repo_ids.append(f'neryotw/bimanual_blue_block_handover_{i}_v30_15d')

    # 19 mobile datasets (skip ID 1 - format issues)
    for i in MOBILE_IDS:
        repo_ids.append(f'neryotw/mobile_bimanual_blue_block_handover_{i}_v30_15d')

    return repo_ids

def main():
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.scripts.train import train
    from lerobot.utils.utils import init_logging

    init_logging()

    repo_ids = build_repo_ids()

    print("=" * 60)
    print("ACT Training - 15D Action Space")
    print("=" * 60)
    print(f"Datasets: {len(repo_ids)}")
    print(f"  - Stationary: {len(STATIONARY_IDS)}")
    print(f"  - Mobile: {len(MOBILE_IDS)}")
    print()

    # Run training via CLI (simpler and handles all the parsing)
    import subprocess
    import json

    repo_ids_str = json.dumps(repo_ids)

    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--dataset.repo_id={repo_ids_str}",
        "--policy.type=act",
        "--policy.chunk_size=100",
        "--policy.n_action_steps=100",
        "--batch_size=8",
        "--steps=100000",
        "--save_checkpoint=true",
        "--save_freq=10000",
        "--output_dir=outputs/act_15d_39datasets",
        "--policy.repo_id=neryotw/act_bimanual_mobile_15d",
        "--policy.push_to_hub=true",
        "--wandb.enable=true",
        "--wandb.project=mimic-act-15d",
    ]

    print("Starting training...")
    print("=" * 60)

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
