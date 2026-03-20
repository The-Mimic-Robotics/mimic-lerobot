import os
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

repo_id = "Mimic-Robotics/mimic_ttt_redx_30hz_x2_ALL"

# The exact mapping from the broken strings to your clean target strings
task_map = {
    "pick red x piece handover place left left": "pick red x handover place top left",
    "pick red x piece handover place bottom left": "pick red x handover place bottom left",
    "pick red x piece handover place bottom middle": "pick red x handover place bottom middle",
    "pick red x piece handover place bottom right": "pick red x handover place bottom right",
    "pick red x piece handover place middle left": "pick red x handover place middle left",
    "pick red x piece handover place middle middle": "pick red x handover place center",
    "pick red x piece handover place middle right": "pick red x handover place middle right",
    "pick red x piece handover place top right": "pick red x handover place top right",
    "pick red x piece handover place top middle": "pick red x handover place top middle"
}

api = HfApi()

print(f"Starting metadata patch for {repo_id}...\n")

try:
    # ==========================================
    # 1. FIX TASKS.PARQUET
    # ==========================================
    print("Fixing tasks.parquet...")
    tasks_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/tasks.parquet")
    df_tasks = pd.read_parquet(tasks_file)

    # Map the DataFrame index (which holds the strings in LeRobot v3.0)
    df_tasks.index = df_tasks.index.map(lambda x: task_map.get(x, x))
    
    local_tasks = "fixed_tasks.parquet"
    df_tasks.to_parquet(local_tasks)

    api.upload_file(
        path_or_fileobj=local_tasks,
        path_in_repo="meta/tasks.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Patched task names in tasks.parquet"
    )
    print(" -> Success: Pushed fixed tasks.parquet")
    os.remove(local_tasks)

    # ==========================================
    # 2. FIX EPISODE CHUNKS
    # ==========================================
    print("\nScanning for episodes.parquet chunks...")
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    ep_files = [f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")]

    # Robust replacement function in case the dataset stores strings in arrays
    def fix_val(val):
        if isinstance(val, str):
            return task_map.get(val, val)
        if isinstance(val, (list, tuple)) or hasattr(val, 'tolist'):
            try:
                lst = val.tolist() if hasattr(val, 'tolist') else list(val)
                return [task_map.get(v, v) if isinstance(v, str) else v for v in lst]
            except Exception:
                pass
        return val

    for ep_file in ep_files:
        local_ep = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=ep_file)
        df_ep = pd.read_parquet(local_ep)
        
        # Apply the fix across all columns just to be safe
        for col in df_ep.columns:
            df_ep[col] = df_ep[col].apply(fix_val)
            
        local_ep_save = "fixed_ep.parquet"
        df_ep.to_parquet(local_ep_save)
        
        api.upload_file(
            path_or_fileobj=local_ep_save,
            path_in_repo=ep_file,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Patched task names in {ep_file}"
        )
        print(f" -> Success: Pushed fixed {ep_file}")
        os.remove(local_ep_save)

    print("\nAll done! Your merged dataset metadata is fully corrected.")

except Exception as e:
    print(f"An error occurred: {e}")