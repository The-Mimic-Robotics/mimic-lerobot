#!/usr/bin/env python3
"""
Convert old bimanual handover datasets (12D actions) to mobile bimanual format (15D actions).

Old format (12D): 6 joints per arm (including gripper) × 2 arms
New format (15D): 6 joints per arm × 2 arms + 3 base velocities (vx, vy, omega)

Strategy: Zero-pad base velocities since old datasets had stationary base.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_dataset(
    input_repo_id: str,
    output_repo_id: str,
    local_dir: Path | None = None,
    push_to_hub: bool = True,
):
    """Convert a v2.1 bimanual dataset to v3.0 mobile bimanual format.
    
    Args:
        input_repo_id: HuggingFace repo ID of old dataset (e.g., 'Mimic-Robotics/bimanual_blue_block_handover_1')
        output_repo_id: HuggingFace repo ID for converted dataset (e.g., 'Mimic-Robotics/mobile_bimanual_blue_block_handover_1')
        local_dir: Local directory for processing (default: /tmp/dataset_conversion)
        push_to_hub: Whether to push converted dataset to HuggingFace Hub
    """
    if local_dir is None:
        local_dir = Path(f"/tmp/dataset_conversion/{output_repo_id.split('/')[-1]}")
    
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting {input_repo_id} -> {output_repo_id}")
    logger.info(f"Working directory: {local_dir}")
    
    # Step 1: Download old dataset metadata
    logger.info("Step 1: Downloading old dataset...")
    old_ds_dir = local_dir / "old_dataset"
    snapshot_download(
        input_repo_id,
        repo_type="dataset",
        local_dir=old_ds_dir,
        revision="v2.1",  # Explicitly use v2.1
    )
    
    # Step 2: Read and update metadata
    logger.info("Step 2: Updating metadata...")
    info_path = old_ds_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    
    # Update codebase version
    info["codebase_version"] = "v3.0"
    info["robot_type"] = "mimic_follower"  # Updated robot type
    
    # Update action feature (12D -> 15D)
    old_action = info["features"]["action"]
    new_action_names = old_action["names"] + ["base_vx", "base_vy", "base_omega"]
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [15],
        "names": new_action_names
    }
    
    # Update observation.state feature (12D -> 15D)
    old_obs_state = info["features"]["observation.state"]
    new_obs_state_names = old_obs_state["names"] + ["base_x", "base_y", "base_theta"]
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [15],
        "names": new_obs_state_names
    }
    
    # Step 3: Create output directory structure
    logger.info("Step 3: Creating output directory...")
    out_dir = local_dir / "converted_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    
    # Write updated info.json
    with open(out_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Copy other metadata files
    for meta_file in ["episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"]:
        src = old_ds_dir / "meta" / meta_file
        if src.exists():
            shutil.copy(src, out_dir / "meta" / meta_file)
    
    # Step 4: Convert parquet data files
    logger.info("Step 4: Converting data files...")
    data_dir = old_ds_dir / "data"
    
    for parquet_file in tqdm(list(data_dir.rglob("*.parquet")), desc="Converting parquet files"):
        # Read old data
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Pad action column (12D -> 15D with zeros)
        if "action" in df.columns:
            old_actions = np.stack(df["action"].values)  # Shape: (N, 12)
            # Add 3 zeros for base velocities (vx, vy, omega)
            zero_padding = np.zeros((old_actions.shape[0], 3), dtype=np.float32)
            new_actions = np.concatenate([old_actions, zero_padding], axis=1)  # Shape: (N, 15)
            df["action"] = list(new_actions)
        
        # Pad observation.state column (12D -> 15D with zeros)
        if "observation.state" in df.columns:
            old_obs = np.stack(df["observation.state"].values)  # Shape: (N, 12)
            # Add 3 zeros for base position (x, y, theta)
            zero_padding = np.zeros((old_obs.shape[0], 3), dtype=np.float32)
            new_obs = np.concatenate([old_obs, zero_padding], axis=1)  # Shape: (N, 15)
            df["observation.state"] = list(new_obs)
        
        # Write converted data
        relative_path = parquet_file.relative_to(data_dir)
        out_file = out_dir / "data" / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert back to pyarrow table and write
        new_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(new_table, out_file, compression="snappy")
    
    # Step 5: Copy video files
    logger.info("Step 5: Copying video files...")
    video_dir = old_ds_dir / "videos"
    if video_dir.exists():
        for video_file in tqdm(list(video_dir.rglob("*.mp4")), desc="Copying videos"):
            relative_path = video_file.relative_to(video_dir)
            out_file = out_dir / "videos" / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(video_file, out_file)
    
    # Step 6: Push to HuggingFace Hub
    if push_to_hub:
        logger.info(f"Step 6: Pushing to HuggingFace Hub ({output_repo_id})...")
        api = HfApi()
        api.upload_folder(
            folder_path=out_dir,
            repo_id=output_repo_id,
            repo_type="dataset",
            commit_message=f"Converted from {input_repo_id} (v2.1 -> v3.0, 12D -> 15D actions)",
        )
        logger.info(f"✅ Dataset uploaded to {output_repo_id}")
    else:
        logger.info(f"✅ Converted dataset saved to {out_dir}")
    
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert old bimanual handover datasets to mobile bimanual format"
    )
    parser.add_argument(
        "--input-repo-id",
        type=str,
        required=True,
        help="Input dataset repo ID (e.g., Mimic-Robotics/bimanual_blue_block_handover_1)",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        required=True,
        help="Output dataset repo ID (e.g., Mimic-Robotics/mobile_bimanual_blue_block_handover_1)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory for processing (default: /tmp/dataset_conversion/...)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push to HuggingFace Hub (only convert locally)",
    )
    
    args = parser.parse_args()
    
    convert_dataset(
        input_repo_id=args.input_repo_id,
        output_repo_id=args.output_repo_id,
        local_dir=Path(args.local_dir) if args.local_dir else None,
        push_to_hub=not args.no_push,
    )


if __name__ == "__main__":
    main()
