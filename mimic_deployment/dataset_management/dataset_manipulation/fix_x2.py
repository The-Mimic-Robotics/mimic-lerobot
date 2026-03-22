import os
import json
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

# --- POINT THIS AT YOUR FRESH DUPLICATE ---
repo_id = "Mimic-Robotics/mimic_ttt_redx_15hz_x2d2"

# The master map of messy strings to clean strings
task_map = {
    "pick red x piece handover place left left": "pick red x handover place top left",
    "pick red x piece handover place bottom middle": "pick red x handover place bottom middle",
    "pick red x piece handover place bottom left": "pick red x handover place bottom left",
    "pick red x piece handover place middle left": "pick red x handover place middle left",
    "pick red x piece handover place bottom right": "pick red x handover place bottom right",
    "pick red x piece handover place top right": "pick red x handover place top right",
    "pick red x piece handover place middle right": "pick red x handover place middle right",
    "pick red x piece handover place top middle": "pick red x handover place top middle",
    "pick red x piece handover place middle middle": "pick red x handover place center",
    
    # Adding the clean versions so they map to themselves
    "pick red x handover place top left": "pick red x handover place top left",
    "pick red x handover place bottom middle": "pick red x handover place bottom middle",
    "pick red x handover place bottom left": "pick red x handover place bottom left",
    "pick red x handover place middle left": "pick red x handover place middle left",
    "pick red x handover place bottom right": "pick red x handover place bottom right",
    "pick red x handover place top right": "pick red x handover place top right",
    "pick red x handover place middle right": "pick red x handover place middle right",
    "pick red x handover place top middle": "pick red x handover place top middle",
    "pick red x handover place center": "pick red x handover place center",
}

api = HfApi()

try:
    print(f"Starting synchronization for {repo_id}...")
    
    # 1. Establish the universal mapping
    unique_clean_tasks = sorted(list(set(task_map.values())))
    clean_to_idx = {task_name: idx for idx, task_name in enumerate(unique_clean_tasks)}
    print(f"Universal Mapping Established: {clean_to_idx}\n")

    # 2. Fetch the old mapping so we can translate old integers
    print("0/3 Backing up old task mapping...")
    try:
        old_tasks_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/tasks.parquet")
        df_old_tasks = pd.read_parquet(old_tasks_path)
        old_idx_to_str = {row['task_index']: name for name, row in df_old_tasks.iterrows()}
    except Exception as e:
        print(f"Warning: Could not fetch old tasks.parquet. Assuming episode chunks use strings.")
        old_idx_to_str = {}

    # 3. Fix tasks.parquet
    print("1/3 Updating meta/tasks.parquet...")
    df_new_tasks = pd.DataFrame({"task_index": range(len(unique_clean_tasks))}, index=unique_clean_tasks)
    df_new_tasks.index.name = "__index_level_0__"
    
    local_tasks = "fixed_tasks.parquet"
    df_new_tasks.to_parquet(local_tasks)
    api.upload_file(path_or_fileobj=local_tasks, path_in_repo="meta/tasks.parquet", repo_id=repo_id, repo_type="dataset")
    os.remove(local_tasks)

    # 4. Fix info.json (The missing link!)
    print("2/3 Updating meta/info.json...")
    info_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/info.json")
    with open(info_file, "r") as f:
        info_data = json.load(f)
        
    info_data['tasks'] = unique_clean_tasks
    
    local_info = "fixed_info.json"
    with open(local_info, "w") as f:
        json.dump(info_data, f, indent=4)
    api.upload_file(path_or_fileobj=local_info, path_in_repo="meta/info.json", repo_id=repo_id, repo_type="dataset")
    os.remove(local_info)

    # 5. Fix episodes
    print("3/3 Updating meta/episodes/*.parquet chunks...")
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    ep_files = [f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")]

    def process_val(val):
        is_list = False
        raw_val = val
        if hasattr(val, 'tolist'):
            raw_val = val.tolist()
        if isinstance(raw_val, (list, tuple)):
            is_list = True
            raw_val = raw_val[0] if len(raw_val) > 0 else raw_val
            
        # Translate to the clean string whether the current value is a string OR an old integer
        if isinstance(raw_val, str):
            clean_str = task_map.get(raw_val, raw_val)
        elif isinstance(raw_val, (int, float)) or str(raw_val).isdigit():
            old_str = old_idx_to_str.get(int(raw_val), "UNKNOWN")
            clean_str = task_map.get(old_str, old_str)
        else:
            clean_str = "UNKNOWN"
            
        new_idx = clean_to_idx.get(clean_str, 0)
        return [new_idx] if is_list else new_idx

    for ep_file in ep_files:
        local_ep = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=ep_file)
        df_ep = pd.read_parquet(local_ep)
        
        task_col = None
        for col in ['tasks', 'task_index', 'task']:
            if col in df_ep.columns:
                task_col = col
                break
                
        if task_col:
            df_ep[task_col] = df_ep[task_col].apply(process_val)
            local_ep_save = "fixed_ep.parquet"
            df_ep.to_parquet(local_ep_save)
            api.upload_file(path_or_fileobj=local_ep_save, path_in_repo=ep_file, repo_id=repo_id, repo_type="dataset")
            os.remove(local_ep_save)
            print(f" -> Cleaned {ep_file}")
            
    print("\nAll done! The dataset is fully synchronized across tasks, info, and episodes.")

except Exception as e:
    print(f"An error occurred: {e}")