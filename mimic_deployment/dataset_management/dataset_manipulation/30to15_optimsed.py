import torch
import shutil
import time
from pathlib import Path
from huggingface_hub import delete_repo
from lerobot.datasets.lerobot_dataset import LeRobotDataset

orig_repo_id = "Mimic-Robotics/mimic_ttt_redBalanced_30hz"
new_repo_id = "Mimic-Robotics/mimic_ttt_redx_15hz_x1"

# nohup python 30to15.py > conversion_log.txt 2>&1 &

# --- 1. CLEANUP PREVIOUS ATTEMPTS ---
local_cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / new_repo_id
if local_cache_dir.exists():
    print(f"Deleting local cache: {local_cache_dir}", flush=True)
    shutil.rmtree(local_cache_dir)

try:
    delete_repo(repo_id=new_repo_id, repo_type="dataset")
    print(f"Deleted existing remote repository: {new_repo_id}", flush=True)
except Exception:
    pass 

# --- 2. LOAD AND CREATE ---
print(f"Loading original dataset: {orig_repo_id}", flush=True)
orig_dataset = LeRobotDataset(orig_repo_id, video_backend="pyav")

print(f"Creating new dataset: {new_repo_id}", flush=True)

# THE OPTIMIZATION: Leveraging the Ryzen 9 9950X (32 Threads)
# We allocate 24 threads to video/image writing, leaving 8 threads 
# free for the main Python loop, data reading, and OS background tasks.
new_dataset = LeRobotDataset.create(
    repo_id=new_repo_id,
    fps=15, 
    features=orig_dataset.features,
    robot_type=orig_dataset.meta.robot_type,
    use_videos=True,
    image_writer_processes=0, 
    image_writer_threads=24  
)

episodes_meta = orig_dataset.meta.episodes
keys_to_drop = {"frame_index", "index", "episode_index", "timestamp", "task_index"}

# --- 3. DOWNSAMPLE ---
total_episodes = orig_dataset.num_episodes
print(f"Starting conversion of {total_episodes} episodes...", flush=True)

start_time_total = time.time()

for ep_idx in range(total_episodes):
    ep_start_time = time.time()
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
                if k in keys_to_drop or k not in new_dataset.features:
                    continue
                
                # CRITICAL MEMORY OPTIMIZATION: 
                # Converting PyTorch Tensors to NumPy arrays to prevent memory leaks
                if new_dataset.features[k]["dtype"] in ["image", "video"]:
                    if isinstance(v, torch.Tensor) and v.ndim == 3:
                        v = v.permute(1, 2, 0).numpy()
                else:
                    if isinstance(v, torch.Tensor):
                        v = v.numpy()
                        
                clean_frame[k] = v
            
            clean_frame["task"] = task_str
            new_dataset.add_frame(clean_frame)
            
        frame_counter += 1
        
    new_dataset.save_episode()
    
    # ETA Calculation
    ep_duration = time.time() - ep_start_time
    eps_remaining = total_episodes - (ep_idx + 1)
    eta_minutes = (eps_remaining * ep_duration) / 60
    
    print(f"Completed Episode {ep_idx + 1}/{total_episodes} | "
          f"Time: {ep_duration:.1f}s | "
          f"ETA: {eta_minutes:.1f} minutes", flush=True)

# --- 4. FINALIZE AND UPLOAD ---
print("Finalizing dataset metadata...", flush=True)
new_dataset.finalize()

total_time = (time.time() - start_time_total) / 60
print(f"Done! Dataset downsampled successfully in {total_time:.1f} minutes.", flush=True)

print("Pushing to Hugging Face Hub...", flush=True)
new_dataset.push_to_hub()
print("Upload complete!", flush=True)