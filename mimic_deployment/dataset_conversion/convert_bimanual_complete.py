#!/usr/bin/env python3
"""
Complete Bimanual → Mobile Bimanual Dataset Converter

Converts old bimanual datasets to new mobile bimanual format with:
- v2.1 → v3.0 version upgrade
- 12D → 15D action/observation expansion (zero-padded base)
- Camera renaming: wrist_right→right_wrist, wrist_left→left_wrist, realsense_top→top
- Dummy front camera creation (blank videos)
- Top camera resolution: 640x480 → 1280x720 (letterboxed)

Usage:
    python convert_bimanual_complete.py \
        --input-repo "Mimic-Robotics/bimanual_blue_block_handover_1" \
        --output-repo "Mimic-Robotics/achal_mobile_bimanual_1" \
        [--no-push] [--keep-temp]
"""

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_blank_video(output_path: Path, duration_sec: float, width: int = 640, height: int = 480, fps: int = 30):
    """Create a blank/black video with specified duration and resolution using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use h264 for speed (LeRobot supports it)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration_sec}:r={fps}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def letterbox_video(input_path: Path, output_path: Path, target_width: int = 1280, target_height: int = 720):
    """
    Letterbox video from 640x480 to 1280x720 by adding black bars using ffmpeg.
    Maintains aspect ratio and centers the original video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use h264 for speed (LeRobot supports it)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"scale=-1:{target_height},pad={target_width}:{target_height}:(ow-iw)/2:0:black",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def convert_v21_to_v30(repo_id: str, local_dir: Path, push_to_hub: bool = False):
    """Run LeRobot's official v2.1 → v3.0 converter with file-based consolidation."""
    logger.info("Step 1: Converting v2.1 → v3.0 format (file-based like mimic_displacement_*)...")
    
    # Use 1MB threshold to consolidate small episodes (25 episodes × 60KB = 1.5MB)
    # This ensures consolidation happens even for datasets with few/small episodes
    cmd = [
        "python", "-m", "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
        f"--root={local_dir.parent}",  # Parent directory containing the repo_id folder
        f"--data-file-size-in-mb=1",  # Lower threshold for small datasets
        f"--video-file-size-in-mb=10",  # Lower threshold for small datasets
        f"--push-to-hub={'true' if push_to_hub else 'false'}"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"v2.1→v3.0 conversion failed:\n{result.stderr}")
        logger.error(f"stdout:\n{result.stdout}")
        raise RuntimeError("Version conversion failed")
    
    logger.info("Version conversion complete")


def expand_actions_and_observations(data_dir: Path):
    """Expand action and observation.state from 12D to 15D by zero-padding."""
    logger.info("Step 2: Expanding actions/observations 12D → 15D...")
    
    parquet_files = list(data_dir.rglob("*.parquet"))
    
    for pq_file in tqdm(parquet_files, desc="Expanding dimensions"):
        # Read parquet
        table = pq.read_table(pq_file)
        df = table.to_pandas()
        
        # Expand action (12D → 15D)
        if "action" in df.columns:
            old_actions = np.stack(df["action"].values)  # (N, 12)
            zero_pad = np.zeros((old_actions.shape[0], 3), dtype=np.float32)
            new_actions = np.concatenate([old_actions, zero_pad], axis=1)  # (N, 15)
            df["action"] = list(new_actions)
        
        # Expand observation.state (12D → 15D)
        if "observation.state" in df.columns:
            old_obs = np.stack(df["observation.state"].values)  # (N, 12)
            zero_pad = np.zeros((old_obs.shape[0], 3), dtype=np.float32)
            new_obs = np.concatenate([old_obs, zero_pad], axis=1)  # (N, 15)
            df["observation.state"] = list(new_obs)
        
        # Write back
        new_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(new_table, pq_file, compression="snappy")
    
    logger.info("Dimension expansion complete")


def rename_and_process_videos(video_dir: Path):
    """
    Rename cameras and process videos:
    - wrist_right → right_wrist (copy as-is)
    - wrist_left → left_wrist (copy as-is)
    - realsense_top → top (letterbox 640x480 → 1280x720)
    - Create dummy front camera (blank videos)
    
    After v2.1→v3.0 conversion, structure is:
    videos/chunk-000/observation.images.{camera}/episode_*.mp4
    
    We reorganize to:
    videos/observation.images.{new_camera}/chunk-000/episode_*.mp4
    """
    logger.info("Step 3: Renaming cameras and processing videos...")
    
    # Find chunk directories
    chunk_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")])
    
    if not chunk_dirs:
        logger.warning("   No chunk directories found - skipping video processing")
        return
    
    logger.info(f"   Found {len(chunk_dirs)} chunk directories")
    
    # Process each chunk (usually just chunk-000)
    for chunk_dir in chunk_dirs:
        # Find camera directories inside this chunk
        cam_dirs_in_chunk = sorted([d for d in chunk_dir.iterdir() if d.is_dir() and d.name.startswith("observation.images")])
        
        for cam_dir in cam_dirs_in_chunk:
            cam_name = cam_dir.name.replace("observation.images.", "")
            
            # Determine new camera name
            if cam_name == "wrist_right":
                new_cam_name = "observation.images.right_wrist"
                logger.info(f"  Processing {cam_name} → right_wrist")
                new_cam_root = video_dir / new_cam_name
                new_cam_root.mkdir(parents=True, exist_ok=True)
                new_chunk = new_cam_root / chunk_dir.name
                new_chunk.mkdir(parents=True, exist_ok=True)
                
                # Move videos as-is
                for video in cam_dir.glob("*.mp4"):
                    shutil.move(str(video), str(new_chunk / video.name))
                    
            elif cam_name == "wrist_left":
                new_cam_name = "observation.images.left_wrist"
                logger.info(f"  Processing {cam_name} → left_wrist")
                new_cam_root = video_dir / new_cam_name
                new_cam_root.mkdir(parents=True, exist_ok=True)
                new_chunk = new_cam_root / chunk_dir.name
                new_chunk.mkdir(parents=True, exist_ok=True)
                
                # Move videos as-is
                for video in cam_dir.glob("*.mp4"):
                    shutil.move(str(video), str(new_chunk / video.name))
                    
            elif cam_name == "realsense_top":
                new_cam_name = "observation.images.top"
                logger.info(f"  Processing {cam_name} → top (letterboxing 640x480 → 1280x720)")
                new_cam_root = video_dir / new_cam_name
                new_cam_root.mkdir(parents=True, exist_ok=True)
                new_chunk = new_cam_root / chunk_dir.name
                new_chunk.mkdir(parents=True, exist_ok=True)
                
                # Letterbox videos
                videos = list(cam_dir.glob("*.mp4"))
                for video in tqdm(videos, desc=f"  Letterboxing {chunk_dir.name}/top"):
                    letterbox_video(video, new_chunk / video.name, target_width=1280, target_height=720)
        
        # Remove old chunk directory after processing
        shutil.rmtree(chunk_dir)
    
    # Create dummy front camera
    logger.info("  Creating front camera (blank videos)")
    front_root = video_dir / "observation.images.front"
    front_root.mkdir(parents=True, exist_ok=True)
    
    # Use right_wrist as reference
    ref_cam_root = video_dir / "observation.images.right_wrist"
    if ref_cam_root.exists():
        for chunk_dir in sorted(ref_cam_root.glob("chunk-*")):
            front_chunk = front_root / chunk_dir.name
            front_chunk.mkdir(parents=True, exist_ok=True)
            
            videos = list(chunk_dir.glob("*.mp4"))
            for ref_video in tqdm(videos, desc=f"  Creating {chunk_dir.name}/front"):
                front_video = front_chunk / ref_video.name
                duration = get_video_duration(ref_video)
                create_blank_video(front_video, duration, width=640, height=480, fps=30)
    
    logger.info("Video processing complete")


def update_metadata(meta_dir: Path):
    """Update info.json with new dimensions, camera names, and robot type."""
    logger.info("Step 4: Updating metadata...")
    
    info_path = meta_dir / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Update version (should be v3.0 after conversion, but double-check)
    info["codebase_version"] = "v3.0"
    
    # Update robot type
    info["robot_type"] = "mimic_follower"
    
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
    
    # Update observation.state feature
    info["features"]["observation.state"]["shape"] = [15]
    info["features"]["observation.state"]["names"] = [
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
        "base_x",
        "base_y",
        "base_theta"
    ]
    
    # Rename camera features
    features = info["features"]
    
    # Remove old camera names
    old_cameras = ["observation.images.wrist_right", "observation.images.wrist_left", "observation.images.realsense_top"]
    for old_cam in old_cameras:
        if old_cam in features:
            del features[old_cam]
    
    # Add new camera features
    # Wrist cameras (640x480)
    for wrist_cam in ["right_wrist", "left_wrist"]:
        features[f"observation.images.{wrist_cam}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False
            }
        }
    
    # Front camera (640x480, dummy)
    features["observation.images.front"] = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30,
            "video.channels": 3,
            "has_audio": False
        }
    }
    
    # Top camera (1280x720, letterboxed)
    features["observation.images.top"] = {
        "dtype": "video",
        "shape": [720, 1280, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 720,
            "video.width": 1280,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30,
            "video.channels": 3,
            "has_audio": False
        }
    }
    
    # Write updated info
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info("Metadata updated")


def convert_dataset(
    input_repo_id: str,
    output_repo_id: str,
    push_to_hub: bool = True,
    keep_temp: bool = False
):
    """
    Complete dataset conversion pipeline.
    
    Args:
        input_repo_id: Old dataset repo (e.g., "Mimic-Robotics/bimanual_blue_block_handover_1")
        output_repo_id: New dataset repo (e.g., "Mimic-Robotics/achal_mobile_bimanual_1")
        push_to_hub: Whether to push to HuggingFace Hub
        keep_temp: Whether to keep temporary working directory
    """
    logger.info("="*70)
    logger.info(f"Converting: {input_repo_id}")
    logger.info(f"       To: {output_repo_id}")
    logger.info("="*70)
    
    # Create temporary working directory
    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_conversion_"))
    logger.info(f"Working directory: {temp_dir}")
    
    try:
        # Download old dataset - must match converter's expected structure: root/{repo_id}/
        logger.info("Downloading source dataset...")
        old_dir = temp_dir / input_repo_id  # Use full repo path
        snapshot_download(
            repo_id=input_repo_id,
            repo_type="dataset",
            local_dir=old_dir,
        )
        
        # Step 1: Convert v2.1 → v3.0 (converter operates on temp_dir as root)
        convert_v21_to_v30(input_repo_id, old_dir, push_to_hub=False)
        
        # Step 2: Expand action/observation dimensions
        data_dir = old_dir / "data"
        expand_actions_and_observations(data_dir)
        
        # Step 3: Rename cameras and process videos
        video_dir = old_dir / "videos"
        rename_and_process_videos(video_dir)
        
        # Step 4: Update metadata
        meta_dir = old_dir / "meta"
        update_metadata(meta_dir)
        
        # Step 5: Push to Hub
        if push_to_hub:
            logger.info(f"Step 5: Pushing to HuggingFace Hub ({output_repo_id})...")
            api = HfApi()
            api.upload_folder(
                folder_path=old_dir,
                repo_id=output_repo_id,
                repo_type="dataset",
                commit_message=f"Converted from {input_repo_id} (v2.1→v3.0, 12D→15D, cameras updated)",
            )
            logger.info(f"Dataset uploaded to {output_repo_id}")
        else:
            logger.info(f"Converted dataset ready at: {old_dir}")
        
        logger.info("="*70)
        logger.info("CONVERSION COMPLETE")
        logger.info("="*70)
        
        return old_dir
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise
    
    finally:
        if not keep_temp and not push_to_hub:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif keep_temp:
            logger.info(f"Temporary directory preserved: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete bimanual → mobile bimanual dataset converter",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input-repo",
        required=True,
        help="Input dataset repo ID (e.g., Mimic-Robotics/bimanual_blue_block_handover_1)"
    )
    parser.add_argument(
        "--output-repo",
        required=True,
        help="Output dataset repo ID (e.g., Mimic-Robotics/achal_mobile_bimanual_1)"
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
    
    convert_dataset(
        input_repo_id=args.input_repo,
        output_repo_id=args.output_repo,
        push_to_hub=not args.no_push,
        keep_temp=args.keep_temp
    )


if __name__ == "__main__":
    main()
