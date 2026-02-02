#!/usr/bin/env python3
"""
Batch convert all bimanual datasets and optionally combine them into one dataset.

Usage:
    # Convert all 21 datasets individually
    python batch_convert_and_combine.py --mode convert --output-dir /tmp/converted_datasets
    
    # Convert and combine into one dataset
    python batch_convert_and_combine.py --mode both --output-dir /tmp/converted_datasets --combined-name mobile_bimanual_combined
    
    # Just combine already-converted datasets
    python batch_convert_and_combine.py --mode combine --input-dir /tmp/converted_datasets --combined-name mobile_bimanual_combined
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of all 21 bimanual datasets
BIMANUAL_DATASETS = [
    "Mimic-Robotics/bimanual_blue_block_handover_1",
    "Mimic-Robotics/bimanual_blue_block_handover_2",
    "Mimic-Robotics/bimanual_blue_block_handover_3",
    "Mimic-Robotics/bimanual_blue_block_handover_4",
    "Mimic-Robotics/bimanual_blue_block_handover_5",
    "Mimic-Robotics/bimanual_blue_block_handover_6",
    "Mimic-Robotics/bimanual_blue_block_handover_7",
    "Mimic-Robotics/bimanual_blue_block_handover_14",
    "Mimic-Robotics/bimanual_blue_block_handover_15",
    "Mimic-Robotics/bimanual_blue_block_handover_16",
    "Mimic-Robotics/bimanual_blue_block_handover_17",
    "Mimic-Robotics/bimanual_blue_block_handover_18",
    "Mimic-Robotics/bimanual_blue_block_handover_19",
    "Mimic-Robotics/bimanual_blue_block_handover_20",
    "Mimic-Robotics/bimanual_blue_block_handover_21",
    "Mimic-Robotics/bimanual_blue_block_handover_22",
    "Mimic-Robotics/bimanual_blue_block_handover_23",
    "Mimic-Robotics/bimanual_blue_block_handover_24",
    "Mimic-Robotics/bimanual_blue_block_handover_25",
    "Mimic-Robotics/bimanual_blue_block_handover_26",
    "Mimic-Robotics/jetson_bimanual_recording_test",
]


def convert_single_dataset(repo_id: str, output_dir: Path) -> Path:
    """Convert a single bimanual dataset using convert_bimanual_FIXED.py."""
    logger.info(f"Converting {repo_id}...")
    
    # Get the dataset name from repo_id
    dataset_name = repo_id.split("/")[-1]
    new_name = f"mobile_{dataset_name}"
    
    # Run the converter
    converter_script = Path(__file__).parent / "convert_bimanual_FIXED.py"
    cmd = [
        sys.executable,
        str(converter_script),
        "--input-repo", repo_id,
        "--output-dir", str(output_dir),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        # The converter outputs to output_dir/org/dataset_name
        converted_path = output_dir / repo_id
        if not converted_path.exists():
            # Try without org prefix
            converted_path = output_dir / dataset_name
        
        logger.info(f"✅ Converted {repo_id} -> {converted_path}")
        return converted_path
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to convert {repo_id}: {e}")
        raise


def combine_datasets(dataset_paths: list[Path], output_path: Path, combined_name: str):
    """Combine multiple converted datasets into one large dataset."""
    logger.info(f"Combining {len(dataset_paths)} datasets into {combined_name}...")
    
    output_path = output_path / combined_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "observation.images.top" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "observation.images.left_wrist" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "observation.images.right_wrist" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    # Combine data parquet files
    logger.info("Combining data parquet files...")
    combined_data = []
    combined_episodes = []
    episode_offset = 0
    frame_offset = 0
    
    for dataset_idx, dataset_path in enumerate(tqdm(dataset_paths, desc="Processing datasets")):
        # Read data
        data_dir = dataset_path / "data" / "chunk-000"
        for data_file in sorted(data_dir.glob("file-*.parquet")):
            df = pd.read_parquet(data_file)
            
            # Offset episode indices
            if "episode_index" in df.columns:
                df["episode_index"] = df["episode_index"] + episode_offset
            if "index" in df.columns:
                df["index"] = df["index"] + frame_offset
            
            combined_data.append(df)
        
        # Read episodes metadata
        episodes_dir = dataset_path / "meta" / "episodes" / "chunk-000"
        for ep_file in sorted(episodes_dir.glob("file-*.parquet")):
            df_ep = pd.read_parquet(ep_file)
            
            # Offset indices in episodes metadata
            if "dataset_from_index" in df_ep.columns:
                df_ep["dataset_from_index"] = df_ep["dataset_from_index"] + frame_offset
                df_ep["dataset_to_index"] = df_ep["dataset_to_index"] + frame_offset
            
            combined_episodes.append(df_ep)
        
        # Update offsets
        num_episodes = len(list(episodes_dir.glob("file-*.parquet")))
        num_frames = sum(len(pd.read_parquet(f)) for f in data_dir.glob("file-*.parquet"))
        episode_offset += num_episodes
        frame_offset += num_frames
        
        # Copy videos (rename with dataset index to avoid conflicts)
        for camera in ["top", "left_wrist", "right_wrist", "front"]:
            src_video_dir = dataset_path / "videos" / f"observation.images.{camera}" / "chunk-000"
            dst_video_dir = output_path / "videos" / f"observation.images.{camera}" / "chunk-000"
            
            for video_file in sorted(src_video_dir.glob("file-*.mp4")):
                # Rename to avoid conflicts: dataset_idx_original_name
                new_name = f"ds{dataset_idx:03d}_{video_file.name}"
                (dst_video_dir / new_name).write_bytes(video_file.read_bytes())
    
    # Write combined data
    logger.info("Writing combined data parquet...")
    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df.to_parquet(output_path / "data" / "chunk-000" / "file-000.parquet", index=False)
    
    # Write combined episodes
    logger.info("Writing combined episodes metadata...")
    combined_ep_df = pd.concat(combined_episodes, ignore_index=True)
    combined_ep_df.to_parquet(output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    
    # Copy and merge metadata from first dataset (they should all be the same structure)
    first_dataset = dataset_paths[0]
    
    # Copy info.json
    src_info = first_dataset / "meta" / "info.json"
    dst_info = output_path / "meta" / "info.json"
    with open(src_info) as f:
        info = json.load(f)
    
    # Update total episodes/frames
    info["total_episodes"] = episode_offset
    info["total_frames"] = frame_offset
    
    with open(dst_info, 'w') as f:
        json.dump(info, f, indent=4)
    
    # Copy tasks.parquet
    src_tasks = first_dataset / "meta" / "tasks.parquet"
    dst_tasks = output_path / "meta" / "tasks.parquet"
    if src_tasks.exists():
        dst_tasks.write_bytes(src_tasks.read_bytes())
    
    logger.info(f"✅ Combined dataset created at {output_path}")
    logger.info(f"   Total episodes: {episode_offset}")
    logger.info(f"   Total frames: {frame_offset}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Batch convert and combine bimanual datasets")
    parser.add_argument("--mode", choices=["convert", "combine", "both"], required=True,
                        help="Mode: 'convert' all datasets, 'combine' existing converted datasets, or 'both'")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for converted datasets")
    parser.add_argument("--input-dir", type=Path,
                        help="Directory with already-converted datasets (for combine mode)")
    parser.add_argument("--combined-name", default="mobile_bimanual_combined",
                        help="Name for combined dataset")
    parser.add_argument("--upload-org", 
                        help="Upload to HuggingFace org (e.g., 'Mimic-Robotics' or 'ac-pate')")
    parser.add_argument("--datasets", nargs="+",
                        help="Specific datasets to convert (default: all 21)")
    
    args = parser.parse_args()
    
    datasets_to_process = args.datasets if args.datasets else BIMANUAL_DATASETS
    
    converted_paths = []
    
    # Convert datasets
    if args.mode in ["convert", "both"]:
        logger.info(f"Converting {len(datasets_to_process)} datasets...")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        for repo_id in datasets_to_process:
            try:
                converted_path = convert_single_dataset(repo_id, args.output_dir)
                converted_paths.append(converted_path)
            except Exception as e:
                logger.error(f"Skipping {repo_id} due to error: {e}")
                continue
    
    # Combine datasets
    if args.mode in ["combine", "both"]:
        if args.mode == "combine":
            # Load already-converted datasets
            input_dir = args.input_dir or args.output_dir
            converted_paths = sorted(input_dir.glob("*/"))
            logger.info(f"Found {len(converted_paths)} datasets to combine in {input_dir}")
        
        if not converted_paths:
            logger.error("No datasets to combine!")
            return
        
        combined_path = combine_datasets(converted_paths, args.output_dir, args.combined_name)
        
        # Upload combined dataset
        if args.upload_org:
            logger.info(f"Uploading combined dataset to {args.upload_org}...")
            api = HfApi()
            repo_id = f"{args.upload_org}/{args.combined_name}"
            
            try:
                api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
                api.upload_folder(
                    folder_path=combined_path,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
                api.create_tag(repo_id, tag="v3.0", repo_type="dataset")
                logger.info(f"✅ Uploaded to {repo_id}")
            except Exception as e:
                logger.error(f"❌ Upload failed: {e}")


if __name__ == "__main__":
    main()
