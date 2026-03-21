#!/usr/bin/env python3
"""
Consolidate all old bimanual datasets into ONE large mobile_bimanual dataset.

This script:
1. Downloads all 21 old datasets
2. Merges them into a single combined dataset directory
3. Runs the conversion pipeline once on the combined dataset
4. Uploads as ONE consolidated dataset to HuggingFace

Usage:
    python3 consolidate_all_datasets.py \
        --output-repo="Mimic-Robotics/mobile_bimanual_blue_block_handover_complete" \
        [--no-push] [--keep-temp]
"""

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from convert_bimanual_complete import (
    convert_v21_to_v30,
    expand_actions_and_observations,
    rename_and_process_videos,
    update_metadata,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All old datasets to consolidate
OLD_DATASETS = [
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


def merge_dataset_into(source_dir: Path, target_dir: Path, episode_offset: int) -> int:
    """
    Merge a single dataset into the target combined dataset.
    
    Args:
        source_dir: Source dataset directory (v2.1 format)
        target_dir: Target combined dataset directory
        episode_offset: Starting episode index for this dataset
    
    Returns:
        Number of episodes added
    """
    logger.info(f"  Merging {source_dir.name} (offset={episode_offset})")
    
    # Create target structure if needed
    (target_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (target_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (target_dir / "meta").mkdir(parents=True, exist_ok=True)
    
    # Copy data files with new episode indices
    src_data = source_dir / "data" / "chunk-000"
    dst_data = target_dir / "data" / "chunk-000"
    
    episode_files = sorted(src_data.glob("episode_*.parquet"))
    for i, ep_file in enumerate(episode_files):
        new_idx = episode_offset + i
        dst_file = dst_data / f"episode_{new_idx:06d}.parquet"
        shutil.copy2(ep_file, dst_file)
    
    # Copy video files with new episode indices
    src_videos = source_dir / "videos" / "chunk-000"
    dst_videos = target_dir / "videos" / "chunk-000"
    
    for camera_dir in src_videos.iterdir():
        if camera_dir.is_dir():
            # Create camera dir in target
            dst_cam_dir = dst_videos / camera_dir.name
            dst_cam_dir.mkdir(exist_ok=True)
            
            # Copy and rename videos
            video_files = sorted(camera_dir.glob("episode_*.mp4"))
            for i, vid_file in enumerate(video_files):
                new_idx = episode_offset + i
                dst_file = dst_cam_dir / f"episode_{new_idx:06d}.mp4"
                shutil.copy2(vid_file, dst_file)
    
    # Append episode metadata
    src_episodes = source_dir / "meta" / "episodes.jsonl"
    dst_episodes = target_dir / "meta" / "episodes.jsonl"
    
    with open(src_episodes, 'r') as src_f:
        episodes = [json.loads(line) for line in src_f]
        
        with open(dst_episodes, 'a') as dst_f:
            for i, ep in enumerate(episodes):
                ep['episode_index'] = episode_offset + i
                dst_f.write(json.dumps(ep) + '\n')
    
    # Copy other metadata on first merge
    if episode_offset == 0:
        for meta_file in ["info.json", "tasks.jsonl", "episodes_stats.jsonl"]:
            src_file = source_dir / "meta" / meta_file
            dst_file = target_dir / "meta" / meta_file
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
    
    return len(episode_files)


def consolidate_all_datasets(output_repo_id: str, push_to_hub: bool = True, keep_temp: bool = False):
    """Consolidate all old datasets into one large dataset."""
    logger.info("="*70)
    logger.info("CONSOLIDATING ALL OLD DATASETS INTO ONE")
    logger.info(f"Output: {output_repo_id}")
    logger.info(f"Total datasets to merge: {len(OLD_DATASETS)}")
    logger.info("="*70)
    
    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_consolidation_"))
    logger.info(f"Working directory: {temp_dir}")
    
    try:
        # Create combined dataset directory
        combined_dir = temp_dir / "combined_bimanual"
        combined_dir.mkdir(parents=True)
        
        # Download and merge all datasets
        episode_offset = 0
        for i, dataset_repo in enumerate(OLD_DATASETS, 1):
            logger.info(f"\n[{i}/{len(OLD_DATASETS)}] Processing {dataset_repo}")
            
            # Download dataset
            dataset_dir = temp_dir / f"dataset_{i}"
            logger.info(f"  Downloading...")
            snapshot_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                local_dir=dataset_dir,
            )
            
            # Merge into combined dataset
            num_episodes = merge_dataset_into(dataset_dir, combined_dir, episode_offset)
            episode_offset += num_episodes
            logger.info(f"  Added {num_episodes} episodes (total now: {episode_offset})")
            
            # Clean up individual dataset to save space
            shutil.rmtree(dataset_dir)
        
        logger.info(f"\n✓ Merged {len(OLD_DATASETS)} datasets into one ({episode_offset} total episodes)")
        
        # Update metadata to reflect combined dataset
        logger.info("\nUpdating combined dataset metadata...")
        info_path = combined_dir / "meta" / "info.json"
        with open(info_path, 'r') as f:
            info = json.load(f)
        info['total_episodes'] = episode_offset
        info['total_chunks'] = 1  # All in chunk-000 for now
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        # Now run the conversion pipeline on the combined dataset
        logger.info("\n" + "="*70)
        logger.info("CONVERTING COMBINED DATASET")
        logger.info("="*70)
        
        # Create a fake repo structure for the converter
        fake_repo_dir = temp_dir / "Mimic-Robotics" / "combined_bimanual_temp"
        fake_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(combined_dir), str(fake_repo_dir))
        
        # Run v2.1 → v3.0 conversion (will consolidate due to large size)
        convert_v21_to_v30("Mimic-Robotics/combined_bimanual_temp", fake_repo_dir, push_to_hub=False)
        
        # Expand dimensions
        data_dir = fake_repo_dir / "data"
        expand_actions_and_observations(data_dir)
        
        # Process videos
        video_dir = fake_repo_dir / "videos"
        rename_and_process_videos(video_dir)
        
        # Update metadata
        meta_dir = fake_repo_dir / "meta"
        update_metadata(meta_dir)
        
        logger.info("\n✓ Conversion complete")
        
        # Push to Hub if requested
        if push_to_hub:
            logger.info(f"\nPushing to HuggingFace Hub ({output_repo_id})...")
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=fake_repo_dir,
                repo_id=output_repo_id,
                repo_type="dataset",
                commit_message=f"Consolidated {len(OLD_DATASETS)} bimanual datasets into one mobile_bimanual dataset",
            )
            logger.info(f"✓ Dataset uploaded to {output_repo_id}")
        else:
            logger.info(f"\n✓ Converted dataset ready at: {fake_repo_dir}")
        
        logger.info("\n" + "="*70)
        logger.info("CONSOLIDATION COMPLETE")
        logger.info("="*70)
        
        return fake_repo_dir
        
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        raise
    
    finally:
        if not keep_temp and not push_to_hub:
            logger.info(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif keep_temp:
            logger.info(f"\nTemporary directory preserved: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate all old bimanual datasets into ONE large mobile_bimanual dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output-repo",
        required=True,
        help="Output consolidated dataset repo ID (e.g., Mimic-Robotics/mobile_bimanual_blue_block_handover_complete)"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push to HuggingFace Hub (only convert locally)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary working directory after conversion"
    )
    
    args = parser.parse_args()
    
    consolidate_all_datasets(
        output_repo_id=args.output_repo,
        push_to_hub=not args.no_push,
        keep_temp=args.keep_temp
    )


if __name__ == "__main__":
    main()
