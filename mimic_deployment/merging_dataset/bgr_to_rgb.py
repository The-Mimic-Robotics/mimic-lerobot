import shutil
import time
import sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


#nohup python mimic_deployment/merging_dataset/bgr_to_rgb.py > conversion_progress.log 2>&1 &


def fix_bgr_to_rgb(src_repo_id: str, dst_repo_id: str, target_camera: str):
    print(f"Loading source dataset: {src_repo_id}", flush=True)
    
    # Use pyav to bypass the local file opener bug
    src_dataset = LeRobotDataset(src_repo_id, video_backend="pyav")
    
    fps = src_dataset.fps
    features = src_dataset.features
    robot_type = src_dataset.meta.robot_type
    total_episodes = src_dataset.num_episodes
    
    dst_path = src_dataset.root.parent / dst_repo_id.split("/")[-1]
    if dst_path.exists():
        print(f"Removing existing output directory: {dst_path}", flush=True)
        shutil.rmtree(dst_path)

    print(f"Creating destination dataset: {dst_repo_id}", flush=True)
    dst_dataset = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        use_videos=True
    )

    print(f"Starting conversion of {total_episodes} episodes...", flush=True)

    for ep_idx in range(total_episodes):
        ep_start_time = time.time()
        
        ep_meta = src_dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        
        # Inner loop without tqdm for clean background logging
        for frame_idx in range(from_idx, to_idx):
            item = src_dataset[frame_idx]
            
            # 1. Flip BGR to RGB on the CHW PyTorch tensor
            if target_camera in item:
                item[target_camera] = item[target_camera][[2, 1, 0], :, :]
            
            # 2. Reconstruct the frame safely
            frame = {}
            for key, ft_schema in features.items():
                if key in ["index", "episode_index", "frame_index", "task_index", "timestamp"]:
                    continue
                
                val = item[key]
                
                # 3. Fix the Shape Mismatch 
                if ft_schema["dtype"] in ["image", "video"]:
                    val = val.permute(1, 2, 0).numpy()
                else:
                    val = val.numpy() 
                    
                frame[key] = val
            
            frame["task"] = item["task"]
            dst_dataset.add_frame(frame)
            
        # Encodes the buffered images into the MP4 chunks
        dst_dataset.save_episode()
        
        # Explicitly wipe the leftover images folder per episode
        images_dir = dst_path / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir, ignore_errors=True)

        # ETA Calculation
        ep_duration = time.time() - ep_start_time
        eps_remaining = total_episodes - (ep_idx + 1)
        eta_minutes = (eps_remaining * ep_duration) / 60
        
        print(f"Completed Episode {ep_idx + 1}/{total_episodes} | "
              f"Time: {ep_duration:.1f}s | "
              f"ETA: {eta_minutes:.1f} minutes", flush=True)

    print("Finalizing dataset metadata...", flush=True)
    dst_dataset.finalize()
    print(f"Done! Fixed dataset saved to {dst_path}", flush=True)

if __name__ == "__main__":
    SOURCE_DATASET = "Mimic-Robotics/mimic_tictactoe_redx_full30hz"
    DESTINATION_DATASET = "Mimic-Robotics/mimic_tictactoe_redx_full30hz_rgb"
    CAMERA_TO_FIX = "observation.images.top"
    
    fix_bgr_to_rgb(SOURCE_DATASET, DESTINATION_DATASET, CAMERA_TO_FIX)