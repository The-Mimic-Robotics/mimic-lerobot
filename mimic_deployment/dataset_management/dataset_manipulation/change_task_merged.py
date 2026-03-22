import os
import json
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

repo_id = "Mimic-Robotics/mimic_ttt_redx_ALLBAL_15hz_stable"

# ... [Keep your existing task_map dictionary here] ...

api = HfApi()
print(f"Starting final string-to-int consolidation for {repo_id}...\n")

try:
    # 1. Create the meta/tasks.jsonl file (v3.0 standard)
    print("Enforcing clean integer mapping in meta/tasks.jsonl...")
    
    unique_clean_tasks = sorted(list(set(task_map.values())))
    clean_to_idx = {task_name: idx for idx, task_name in enumerate(unique_clean_tasks)}
    
    local_tasks_jsonl = "fixed_tasks.jsonl"
    with open(local_tasks_jsonl, "w") as f:
        for clean_name, idx in clean_to_idx.items():
            # LeRobot v3.0 expects JSONL records
            record = {"task_index": idx, "task": clean_name}
            f.write(json.dumps(record) + "\n")
            
    api.upload_file(
        path_or_fileobj=local_tasks_jsonl,
        path_in_repo="meta/tasks.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Forced clean task mapping via tasks.jsonl"
    )
    os.remove(local_tasks_jsonl)
    print(f" -> Mapping established: {clean_to_idx}\n")

    # 2. Fix the episodes chunks by replacing strings with integers
    print("Scanning for episodes.parquet chunks to replace strings with integers...")
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    ep_files = [f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")]

    def process_val(val):
        is_list = False
        raw_val = val
        
        # Aggressive unwrapping
        if hasattr(val, 'tolist'):
            raw_val = val.tolist()
        if isinstance(raw_val, (list, tuple)):
            is_list = True
            raw_val = raw_val[0] if len(raw_val) > 0 else raw_val
            
        if isinstance(raw_val, (int, float)) or (isinstance(raw_val, str) and raw_val.isdigit()):
            return [int(raw_val)] if is_list else int(raw_val)
            
        clean_str = task_map.get(raw_val, raw_val)
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
            # Apply the fix!
            df_ep[task_col] = df_ep[task_col].apply(process_val)
            
            # CRITICAL: Force the column to be an integer type so Parquet doesn't save it as an 'object' (string)
            # If your tasks are stored as single integers (not lists), uncomment the next line:
            # df_ep[task_col] = df_ep[task_col].astype(int)
            
            local_ep_save = "fixed_ep.parquet"
            df_ep.to_parquet(local_ep_save)
            
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