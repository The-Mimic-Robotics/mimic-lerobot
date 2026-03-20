import torch
import shutil
from pathlib import Path
from huggingface_hub import delete_repo
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.datasets.video_utils

# --- THE ABSOLUTE BYPASS MONKEYPATCH ---
_orig_decode = lerobot.datasets.video_utils.decode_video_frames

def _patched_decode(video_path, timestamps, tolerance_s, backend):
    # 1. Unpack fsspec LocalFileOpener to fix TorchCodec crashes
    if hasattr(video_path, "path"):
        video_path = str(video_path.path)
    elif hasattr(video_path, "name"):
        video_path = str(video_path.name)
    else:
        video_path = str(video_path)
        
    # 2. Try routing to TorchCodec first (fastest)
    try:
        import torchcodec
        return _orig_decode(video_path, timestamps, tolerance_s, "torchcodec")
    except ImportError:
        pass # Fall back to our custom pure PyAV implementation
        
    # 3. Pure PyAV implementation (Bypasses Torchvision entirely)
    import av
    import numpy as np
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    
    for ts in timestamps:
        # Seek slightly behind the target timestamp
        seek_ts = max(0.0, ts - 0.5)
        target_pts = int(seek_ts / float(stream.time_base))
        container.seek(target_pts, stream=stream)
        
        closest_frame = None
        min_diff = float("inf")
        
        for frame in container.decode(stream):
            if frame.pts is None: 
                continue
            pts_sec = float(frame.pts * stream.time_base)
            diff = abs(pts_sec - ts)
            
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame
                
            # If we passed the target and the difference is growing, stop searching
            if pts_sec > ts and diff > min_diff:
                break
                
        if closest_frame is not None:
            img = closest_frame.to_ndarray(format="rgb24")
            # Convert to (C, H, W) exactly as LeRobot expects
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            frames.append(tensor)
        else:
            raise RuntimeError(f"Could not find frame for timestamp {ts}")
            
    container.close()
    return torch.stack(frames)

# Inject the patch
lerobot.datasets.video_utils.decode_video_frames = _patched_decode
# ---------------------------------------

orig_repo_id = "Mimic-Robotics/mimic_ttt_redx_full30hz_rgb"
new_repo_id = "Mimic-Robotics/mimic_ttt_red_15hz"

# --- 1. CLEANUP PREVIOUS ATTEMPTS ---
local_cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / new_repo_id
if local_cache_dir.exists():
    print(f"Deleting local cache: {local_cache_dir}")
    shutil.rmtree(local_cache_dir)

try:
    delete_repo(repo_id=new_repo_id, repo_type="dataset")
    print(f"Deleted existing remote repository: {new_repo_id}")
except Exception:
    pass 

# --- 2. LOAD AND CREATE ---
# The backend argument here is now safely ignored by our patch, 
# but kept to satisfy the internal API signature
orig_dataset = LeRobotDataset(orig_repo_id, video_backend="pyav")

new_dataset = LeRobotDataset.create(
    repo_id=new_repo_id,
    fps=15, 
    features=orig_dataset.features,
    robot_type=orig_dataset.meta.robot_type,
    use_videos=True,
    video_backend="pyav" 
)

episodes_meta = orig_dataset.meta.episodes
keys_to_drop = {"frame_index", "index", "episode_index", "timestamp", "task_index"}

# --- 3. DOWNSAMPLE ---
for ep_idx in range(orig_dataset.num_episodes):
    start_idx = episodes_meta["dataset_from_index"][ep_idx]
    end_idx = episodes_meta["dataset_to_index"][ep_idx]
    
    # --- ROBUST TASK EXTRACTION ---
    task_str = "default task" 
    if "tasks" in episodes_meta.column_names:
        task_raw = episodes_meta["tasks"][ep_idx]
        
        while True:
            if hasattr(task_raw, "as_py"):
                task_raw = task_raw.as_py()
            elif hasattr(task_raw, "tolist"):
                task_raw = task_raw.tolist()
            elif isinstance(task_raw, (list, tuple)) and len(task_raw) > 0:
                task_raw = task_raw[0]
            else:
                break
                
        if isinstance(task_raw, str):
            task_str = task_raw
        elif hasattr(orig_dataset.meta, "tasks") and orig_dataset.meta.tasks is not None:
            try:
                task_str = orig_dataset.meta.tasks.iloc[int(task_raw)].name
            except Exception:
                task_str = str(task_raw)

    frame_counter = 0
    for i in range(start_idx, end_idx):
        if frame_counter % 2 == 0:
            frame_data = orig_dataset[i]
            clean_frame = {}
            
            for k, v in frame_data.items():
                if k in keys_to_drop:
                    continue
                if k not in new_dataset.features:
                    continue
                if new_dataset.features[k]["dtype"] in ["image", "video"]:
                    if isinstance(v, torch.Tensor) and v.ndim == 3:
                        v = v.permute(1, 2, 0)
                        
                clean_frame[k] = v
            
            clean_frame["task"] = task_str
            new_dataset.add_frame(clean_frame)
            
        frame_counter += 1
        
    new_dataset.save_episode()

# --- 4. FINALIZE AND UPLOAD ---
new_dataset.finalize()
new_dataset.push_to_hub()