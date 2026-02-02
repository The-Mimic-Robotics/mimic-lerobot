#!/usr/bin/env python3
"""
Complete Bimanual → Mobile Bimanual Dataset Converter  - FIXED VERSION

This script:
1. Uses LeRobot's official v2.1→v3.0 converter (creates proper metadata)
2. Expands dimensions 12D→15D for actions/observations  
3. Renames cameras and processes videos
4. Updates metadata

Usage:
    python convert_bimanual_FIXED.py \
        --input-repo "Mimic-Robotics/bimanual_blue_block_handover_1" \
        --output-dir "/tmp/converted_dataset" \
        [--upload-to "Mimic-Robotics/mobile_bimanual_test"]
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def step1_official_v3_conversion(repo_id: str, output_dir: Path):
    """Run LeRobot's official v2.1→v3.0 converter."""
    logger.info(f"STEP 1: Converting {repo_id} to v3.0 format...")
    
    # Use Python script directly
    conv_script = Path(__file__).parent.parent.parent / "src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py"
    
    cmd = [
        sys.executable, str(conv_script),
        f"--repo-id={repo_id}",
        f"--root={output_dir}",
        "--push-to-hub=false",
        "--force-conversion",
        f"--data-file-size-in-mb=100",
        f"--video-file-size-in-mb=500"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Dataset will be at output_dir/repo_id (repo_id includes organization)
    dataset_path = output_dir / repo_id
    logger.info(f"✅ V3.0 conversion complete: {dataset_path}")
    return dataset_path


def step2_expand_dimensions(dataset_path: Path):
    """Expand 12D actions/observations to 15D by zero-padding base dimensions."""
    logger.info("STEP 2: Expanding actions/observations 12D → 15D...")
    
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Process all parquet files in data directory
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    
    for pq_file in tqdm(parquet_files, desc="Expanding dimensions"):
        table = pq.read_table(pq_file)
        
        # Expand action (12D → 15D)
        if "action" in table.column_names:
            action_col = table["action"]
            expanded_actions = []
            
            for i in range(len(action_col)):
                action = action_col[i].as_py()
                if len(action) == 12:
                    # Append [base_vx=0, base_vy=0, base_omega=0]
                    expanded = action + [0.0, 0.0, 0.0]
                    expanded_actions.append(expanded)
                else:
                    expanded_actions.append(action)
            
            # Replace column
            new_action_col = pa.array(expanded_actions, type=pa.list_(pa.float32(), 15))
            table = table.set_column(
                table.schema.get_field_index("action"),
                "action",
                new_action_col
            )
        
        # Expand observation.state (12D → 15D)
        if "observation.state" in table.column_names:
            obs_col = table["observation.state"]
            expanded_obs = []
            
            for i in range(len(obs_col)):
                obs = obs_col[i].as_py()
                if len(obs) == 12:
                    # Append [base_x=0, base_y=0, base_theta=0]
                    expanded = obs + [0.0, 0.0, 0.0]
                    expanded_obs.append(expanded)
                else:
                    expanded_obs.append(obs)
            
            # Replace column
            new_obs_col = pa.array(expanded_obs, type=pa.list_(pa.float32(), 15))
            table = table.set_column(
                table.schema.get_field_index("observation.state"),
                "observation.state",
                new_obs_col
            )
        
        # Write back
        pq.write_table(table, pq_file)
    
    logger.info(f"✅ Expanded {len(parquet_files)} parquet files")


def step3_process_videos(dataset_path: Path):
    """Rename cameras and process videos."""
    logger.info("STEP 3: Processing videos and renaming cameras...")
    
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    # Camera renaming map
    camera_map = {
        "observation.images.realsense_top": "observation.images.top",
        "observation.images.wrist_left": "observation.images.left_wrist",
        "observation.images.wrist_right": "observation.images.right_wrist"
    }
    
    # Process each camera type
    for old_cam, new_cam in camera_map.items():
        old_cam_dir = videos_dir / old_cam
        if not old_cam_dir.exists():
            continue
        
        # Create new camera directory
        new_cam_dir = videos_dir / new_cam
        new_cam_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all chunk subdirectories
        chunk_dirs = sorted([d for d in old_cam_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")])
        
        for chunk_dir in chunk_dirs:
            chunk_name = chunk_dir.name
            new_chunk_dir = new_cam_dir / chunk_name
            new_chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all videos for this chunk
            video_files = sorted(chunk_dir.glob("*.mp4"))
            
            if "top" in new_cam:
                # Letterbox top camera videos 640x480 → 1280x720
                logger.info(f"Letterboxing {len(video_files)} top camera videos in {chunk_name}...")
                for video_file in tqdm(video_files, desc=f"Letterboxing {chunk_name}"):
                    output_file = new_chunk_dir / video_file.name
                    letterbox_video(video_file, output_file)
            else:
                # Just copy wrist cameras
                logger.info(f"Copying {len(video_files)} {new_cam.split('.')[-1]} videos in {chunk_name}...")
                for video_file in tqdm(video_files, desc=f"Copying {chunk_name}"):
                    output_file = new_chunk_dir / video_file.name
                    shutil.copy2(video_file, output_file)
        
        # Remove old camera directory
        shutil.rmtree(old_cam_dir)
    
    # Create blank front camera videos
    logger.info("Creating blank front camera videos...")
    # Use right_wrist as reference
    ref_cam_dir = videos_dir / "observation.images.right_wrist"
    if ref_cam_dir.exists():
        front_cam_dir = videos_dir / "observation.images.front"
        front_cam_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each chunk
        chunk_dirs = sorted([d for d in ref_cam_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")])
        for chunk_dir in chunk_dirs:
            chunk_name = chunk_dir.name
            front_chunk_dir = front_cam_dir / chunk_name
            front_chunk_dir.mkdir(parents=True, exist_ok=True)
            
            ref_videos = sorted(chunk_dir.glob("*.mp4"))
            for ref_video in tqdm(ref_videos, desc=f"Creating front/{chunk_name}"):
                duration = get_video_duration(ref_video)
                output_file = front_chunk_dir / ref_video.name
                create_blank_video(output_file, duration)
    
    logger.info("✅ Video processing complete")


def step4_update_metadata(dataset_path: Path):
    """Update info.json metadata."""
    logger.info("STEP 4: Updating metadata...")
    
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Info file not found: {info_path}")
    
    with open(info_path) as f:
        info = json.load(f)
    
    # Update robot type
    info["robot_type"] = "mimic_follower"
    
    # Update action dimensions and names (15D)
    info["features"]["action"]["shape"] = [15]
    info["features"]["action"]["names"] = [
        "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
        "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
        "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
        "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
        "base_vx", "base_vy", "base_omega"
    ]
    
    # Update observation.state dimensions and names (15D)
    info["features"]["observation.state"]["shape"] = [15]
    info["features"]["observation.state"]["names"] = [
        "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
        "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
        "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
        "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
        "base_x", "base_y", "base_theta"
    ]
    
    # Rename camera features
    if "observation.images.wrist_right" in info["features"]:
        info["features"]["observation.images.right_wrist"] = info["features"].pop("observation.images.wrist_right")
    
    if "observation.images.wrist_left" in info["features"]:
        info["features"]["observation.images.left_wrist"] = info["features"].pop("observation.images.wrist_left")
    
    # Update top camera resolution
    if "observation.images.realsense_top" in info["features"]:
        top_feature = info["features"].pop("observation.images.realsense_top")
        top_feature["shape"] = [720, 1280, 3]
        top_feature["names"] = ["height", "width", "channels"]
        top_feature["info"]["video.height"] = 720
        top_feature["info"]["video.width"] = 1280
        info["features"]["observation.images.top"] = top_feature
    
    # Add front camera feature
    info["features"]["observation.images.front"] = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30,
            "video.channels": 3,
            "has_audio": False
        }
    }
    
    # Write back
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    logger.info("✅ Metadata updated")


def step5_update_episodes_metadata(dataset_path: Path):
    """Update episodes metadata to rename camera columns and add front camera."""
    logger.info("STEP 5: Updating episodes metadata...")
    
    episodes_dir = dataset_path / "meta" / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")
    
    # Camera column mapping (old -> new)
    column_renames = {
        "videos/observation.images.realsense_top/chunk_index": "videos/observation.images.top/chunk_index",
        "videos/observation.images.realsense_top/file_index": "videos/observation.images.top/file_index",
        "videos/observation.images.realsense_top/from_timestamp": "videos/observation.images.top/from_timestamp",
        "videos/observation.images.realsense_top/to_timestamp": "videos/observation.images.top/to_timestamp",
        "videos/observation.images.wrist_left/chunk_index": "videos/observation.images.left_wrist/chunk_index",
        "videos/observation.images.wrist_left/file_index": "videos/observation.images.left_wrist/file_index",
        "videos/observation.images.wrist_left/from_timestamp": "videos/observation.images.left_wrist/from_timestamp",
        "videos/observation.images.wrist_left/to_timestamp": "videos/observation.images.left_wrist/to_timestamp",
        "videos/observation.images.wrist_right/chunk_index": "videos/observation.images.right_wrist/chunk_index",
        "videos/observation.images.wrist_right/file_index": "videos/observation.images.right_wrist/file_index",
        "videos/observation.images.wrist_right/from_timestamp": "videos/observation.images.right_wrist/from_timestamp",
        "videos/observation.images.wrist_right/to_timestamp": "videos/observation.images.right_wrist/to_timestamp",
        "stats/observation.images.realsense_top/min": "stats/observation.images.top/min",
        "stats/observation.images.realsense_top/max": "stats/observation.images.top/max",
        "stats/observation.images.realsense_top/mean": "stats/observation.images.top/mean",
        "stats/observation.images.realsense_top/std": "stats/observation.images.top/std",
        "stats/observation.images.realsense_top/count": "stats/observation.images.top/count",
        "stats/observation.images.wrist_left/min": "stats/observation.images.left_wrist/min",
        "stats/observation.images.wrist_left/max": "stats/observation.images.left_wrist/max",
        "stats/observation.images.wrist_left/mean": "stats/observation.images.left_wrist/mean",
        "stats/observation.images.wrist_left/std": "stats/observation.images.left_wrist/std",
        "stats/observation.images.wrist_left/count": "stats/observation.images.left_wrist/count",
        "stats/observation.images.wrist_right/min": "stats/observation.images.right_wrist/min",
        "stats/observation.images.wrist_right/max": "stats/observation.images.right_wrist/max",
        "stats/observation.images.wrist_right/mean": "stats/observation.images.right_wrist/mean",
        "stats/observation.images.wrist_right/std": "stats/observation.images.right_wrist/std",
        "stats/observation.images.wrist_right/count": "stats/observation.images.right_wrist/count",
    }
    
    # Process all episode parquet files
    episode_files = sorted(episodes_dir.rglob("*.parquet"))
    
    for ep_file in tqdm(episode_files, desc="Updating episodes metadata"):
        df = pd.read_parquet(ep_file)
        
        # Rename columns
        df = df.rename(columns=column_renames)
        
        # Add front camera columns (copy from right_wrist as template)
        if "videos/observation.images.right_wrist/chunk_index" in df.columns:
            df["videos/observation.images.front/chunk_index"] = df["videos/observation.images.right_wrist/chunk_index"]
            df["videos/observation.images.front/file_index"] = df["videos/observation.images.right_wrist/file_index"]
            df["videos/observation.images.front/from_timestamp"] = df["videos/observation.images.right_wrist/from_timestamp"]
            df["videos/observation.images.front/to_timestamp"] = df["videos/observation.images.right_wrist/to_timestamp"]
        
        # Add front camera stats (zeros since it's blank video)
        if "stats/observation.images.right_wrist/count" in df.columns:
            df["stats/observation.images.front/min"] = df["stats/observation.images.right_wrist/min"].apply(lambda x: np.zeros_like(x))
            df["stats/observation.images.front/max"] = df["stats/observation.images.right_wrist/max"].apply(lambda x: np.zeros_like(x))
            df["stats/observation.images.front/mean"] = df["stats/observation.images.right_wrist/mean"].apply(lambda x: np.zeros_like(x))
            df["stats/observation.images.front/std"] = df["stats/observation.images.right_wrist/std"].apply(lambda x: np.zeros_like(x))
            df["stats/observation.images.front/count"] = df["stats/observation.images.right_wrist/count"]
        
        # Write back
        df.to_parquet(ep_file, index=False)
    
    logger.info(f"✅ Updated {len(episode_files)} episode metadata files")


def letterbox_video(input_path: Path, output_path: Path, target_width: int = 1280, target_height: int = 720):
    """Letterbox video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"scale=-1:{target_height},pad={target_width}:{target_height}:(ow-iw)/2:0:black",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def create_blank_video(output_path: Path, duration_sec: float, width: int = 640, height: int = 480, fps: int = 30):
    """Create blank video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration_sec}:r={fps}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Convert bimanual dataset to mobile_bimanual format")
    parser.add_argument("--input-repo", required=True, help="Source HuggingFace repo (e.g., Mimic-Robotics/bimanual_blue_block_handover_1)")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument("--upload-to", help="Upload to HuggingFace repo (optional)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Official v3.0 conversion
        dataset_path = step1_official_v3_conversion(args.input_repo, output_dir)
        
        # Step 2: Expand dimensions
        step2_expand_dimensions(dataset_path)
        
        # Step 3: Process videos
        step3_process_videos(dataset_path)
        
        # Step 4: Update metadata
        step4_update_metadata(dataset_path)
        
        # Step 5: Update episodes metadata
        step5_update_episodes_metadata(dataset_path)
        
        logger.info(f"\n✅ CONVERSION COMPLETE: {dataset_path}")
        
        # Optional: Upload to HuggingFace
        if args.upload_to:
            logger.info(f"\nUploading to {args.upload_to}...")
            api = HfApi()
            api.create_repo(args.upload_to, repo_type="dataset", private=False, exist_ok=True)
            api.upload_folder(
                folder_path=dataset_path,
                repo_id=args.upload_to,
                repo_type="dataset"
            )
            api.create_tag(args.upload_to, tag="v3.0", repo_type="dataset")
            logger.info(f"✅ Uploaded to {args.upload_to}")
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
