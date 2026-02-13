"""
Script to downsample a LeRobot dataset from 30Hz to 15Hz.
STRICT VERSION: Uses an allowlist to prevent 'Extra features' errors.
"""
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tqdm
import torch
import numpy as np

# CONFIGURATION
SOURCE_REPO = "Mimic-Robotics/mimic_tictactoe_redx_full30hz"
TARGET_REPO = "Mimic-Robotics/mimic_tictactoe_redx_full15hz"
TARGET_FPS = 15
DOWNSAMPLE_FACTOR = 2  # 30Hz / 15Hz = 2

def resample_dataset():
    # 1. Load Source
    print(f"Loading source: {SOURCE_REPO}...")
    ds_source = LeRobotDataset(SOURCE_REPO)
    
    # 2. Setup Target Config
    info = ds_source.meta.info
    # source_features contains ONLY the valid data keys (e.g., 'observation.state', 'action')
    # It does NOT contain 'index', 'timestamp', etc.
    source_features = ds_source.features 
    
    use_videos = len(ds_source.meta.video_keys) > 0
    
    print(f"Creating target: {TARGET_REPO} at {TARGET_FPS} fps...")
    ds_target = LeRobotDataset.create(
        repo_id=TARGET_REPO,
        fps=TARGET_FPS,
        features=source_features,
        robot_type=info.get("robot_type", "so101"),
        tolerance_s=info.get("tolerance_s", 0.01),
        use_videos=use_videos
    )

    # 3. Resample Loop
    print(f"Processing {ds_source.num_episodes} episodes...")
    
    for ep_idx in tqdm.tqdm(range(ds_source.num_episodes)):
        # Get range from metadata
        episode_meta = ds_source.meta.episodes[ep_idx]
        start_idx = episode_meta["dataset_from_index"]
        end_idx = episode_meta["dataset_to_index"]
        
        # Iterate frames with a STEP of 2 (Downsampling)
        for frame_idx in range(start_idx, end_idx, DOWNSAMPLE_FACTOR):
            # Load raw frame (contains lots of extra metadata)
            raw_frame = ds_source[frame_idx]
            
            # --- THE FIX: STRICT CONSTRUCTION ---
            # Start with an empty dict and ONLY add what is allowed.
            clean_frame = {}
            
            # A. Add Task (Required)
            if "task" in raw_frame:
                clean_frame["task"] = raw_frame["task"]

            # B. Add Only Valid Features
            for key in source_features:
                if key not in raw_frame:
                    continue
                
                val = raw_frame[key]
                
                # C. Fix Image Shapes (C, H, W) -> (H, W, C)
                # LeRobot returns tensors as (Channels, Height, Width)
                # add_frame expects (Height, Width, Channels)
                if source_features[key]["dtype"] in ["image", "video"]:
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 3 and val.shape[0] in [1, 3]: 
                            val = val.permute(1, 2, 0) 
                    elif isinstance(val, np.ndarray):
                        if val.ndim == 3 and val.shape[0] in [1, 3]:
                            val = np.transpose(val, (1, 2, 0))

                clean_frame[key] = val
            
            # D. Add to buffer
            # Note: We deliberately EXCLUDE 'timestamp'. 
            # ds_target will auto-generate perfect 15Hz timestamps (0.0, 0.066, 0.133...)
            ds_target.add_frame(clean_frame)
            
        # Commit the episode
        ds_target.save_episode()

    # 4. Finalize
    print("Finalizing (writing metadata footer)...")
    ds_target.finalize()
    
    print("Uploading to Hub...")
    ds_target.push_to_hub()
    print("Done! 🚀")

if __name__ == "__main__":
    resample_dataset()