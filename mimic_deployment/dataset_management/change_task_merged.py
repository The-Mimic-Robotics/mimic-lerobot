import os
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

repo_id = "Mimic-Robotics/mimic_ttt_redx_ALLBAL_15hz"

# The map of messy strings to clean strings
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
    
    # Adding the already clean versions just in case it hits one
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

print(f"Starting final string-to-int consolidation for {repo_id}...\n")

try:
    # 1. Ensure tasks.parquet is perfectly formatted and get our integer mapping
    print("Enforcing clean 0-8 integer mapping in tasks.parquet...")
    
    unique_clean_tasks = sorted(list(set(task_map.values())))
    df_new_tasks = pd.DataFrame({"task_index": range(len(unique_clean_tasks))}, index=unique_clean_tasks)
    df_new_tasks.index.name = "__index_level_0__"
    
    local_tasks = "fixed_tasks.parquet"
    df_new_tasks.to_parquet(local_tasks)
    api.upload_file(
        path_or_fileobj=local_tasks,
        path_in_repo="meta/tasks.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Forced clean task integer mapping"
    )
    os.remove(local_tasks)
    
    # Create a dictionary to instantly look up the new integer index from the clean string
    clean_to_idx = {task_name: idx for idx, task_name in enumerate(unique_clean_tasks)}
    print(f" -> Mapping established: {clean_to_idx}\n")

    # 2. Fix the episodes chunks by replacing strings with integers
    print("Scanning for episodes.parquet chunks to replace strings with integers...")
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    ep_files = [f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")]

    # Robust parser to handle strings, lists, or stray integers
    def process_val(val):
        is_list = False
        raw_val = val
        
        # Aggressive unwrapping
        if hasattr(val, 'tolist'):
            raw_val = val.tolist()
        if isinstance(raw_val, (list, tuple)):
            is_list = True
            raw_val = raw_val[0] if len(raw_val) > 0 else raw_val
            
        # If it's already an integer, leave it alone
        if isinstance(raw_val, (int, float)) or (isinstance(raw_val, str) and raw_val.isdigit()):
            return [int(raw_val)] if is_list else int(raw_val)
            
        # Convert raw string -> clean string -> new integer index
        clean_str = task_map.get(raw_val, raw_val)
        new_idx = clean_to_idx.get(clean_str, 0) # defaults to 0 if something goes horribly wrong
        
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
            # Apply the fix!
            df_ep[task_col] = df_ep[task_col].apply(process_val)
            
            local_ep_save = "fixed_ep.parquet"
            df_ep.to_parquet(local_ep_save)
            
            # Push back to hub
            api.upload_file(
                path_or_fileobj=local_ep_save,
                path_in_repo=ep_file,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Converted string tasks to int indices in {ep_file}"
            )
            print(f" -> Success: Mapped strings to integers in {ep_file}")
            os.remove(local_ep_save)
        else:
            print(f" -> Warning: No task column found in {ep_file}")

    print("\nAll done! Your dataset tasks are now natively using LeRobot integer indices.")

except Exception as e:
    print(f"An error occurred: {e}")