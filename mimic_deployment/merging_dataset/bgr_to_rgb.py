import shutil
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def fix_bgr_to_rgb(src_repo_id: str, dst_repo_id: str, target_camera: str):
    print(f"Loading source dataset: {src_repo_id}")
    
    # Use pyav to bypass the local file opener bug
    src_dataset = LeRobotDataset(src_repo_id, video_backend="pyav")
    
    fps = src_dataset.fps
    features = src_dataset.features
    robot_type = src_dataset.meta.robot_type
    
    dst_path = src_dataset.root.parent / dst_repo_id.split("/")[-1]
    if dst_path.exists():
        print(f"Removing existing output directory: {dst_path}")
        shutil.rmtree(dst_path)

    print(f"Creating destination dataset: {dst_repo_id}")
    dst_dataset = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        use_videos=True
    )

    for ep_idx in range(src_dataset.num_episodes):
        ep_meta = src_dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        
        print(f"Processing Episode {ep_idx} (frames {from_idx} to {to_idx - 1})...")
        
        for frame_idx in tqdm(range(from_idx, to_idx), leave=False):
            item = src_dataset[frame_idx]
            
            # 1. Flip BGR to RGB on the CHW PyTorch tensor (Channel 0 and 2 swapped)
            if target_camera in item:
                item[target_camera] = item[target_camera][[2, 1, 0], :, :]
            
            # 2. Reconstruct the frame safely
            frame = {}
            for key, ft_schema in features.items():
                # Strip out tracking variables to avoid 'Extra features' validation error
                if key in ["index", "episode_index", "frame_index", "task_index", "timestamp"]:
                    continue
                
                val = item[key]
                
                # 3. Fix the Shape Mismatch 
                # Convert from PyTorch CHW back to numpy HWC before saving
                if ft_schema["dtype"] in ["image", "video"]:
                    val = val.permute(1, 2, 0).numpy()
                else:
                    # Ensure states and actions are returned to numpy arrays
                    val = val.numpy() 
                    
                frame[key] = val
            
            frame["task"] = item["task"]
            dst_dataset.add_frame(frame)
            
        # This encodes the buffered images into the MP4 chunks
        dst_dataset.save_episode()
        
        # --- THE FIX ---
        # Explicitly wipe the leftover images folder per episode to prevent drive bloat
        images_dir = dst_path / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir, ignore_errors=True)

    print("Finalizing dataset metadata...")
    dst_dataset.finalize()
    print(f"Done! Fixed dataset saved to {dst_path}")

if __name__ == "__main__":
    SOURCE_DATASET = "Mimic-Robotics/test2"
    DESTINATION_DATASET = "Mimic-Robotics/test2_fixed"
    CAMERA_TO_FIX = "observation.images.top"
    
    fix_bgr_to_rgb(SOURCE_DATASET, DESTINATION_DATASET, CAMERA_TO_FIX)