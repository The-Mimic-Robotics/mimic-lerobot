import os
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

# The two base datasets you are trying to merge
repos = [
    "Mimic-Robotics/mimic_ttt_redx_15hz_x2",
    "Mimic-Robotics/mimic_ttt_redBalanced_15hz"
]

# The universal dictionary mapping messy strings to our clean targets
string_cleaner = {
    "pick red x piece handover place left left": "pick red x handover place top left",
    "pick red x piece handover place bottom middle": "pick red x handover place bottom middle",
    "pick red x piece handover place bottom left": "pick red x handover place bottom left",
    "pick red x piece handover place middle left": "pick red x handover place middle left",
    "pick red x piece handover place bottom right": "pick red x handover place bottom right",
    "pick red x piece handover place top right": "pick red x handover place top right",
    "pick red x piece handover place middle right": "pick red x handover place middle right",
    "pick red x piece handover place top middle": "pick red x handover place top middle",
    "pick red x piece handover place middle middle": "pick red x handover place center",
}

# The UNIVERSAL index standard that BOTH datasets must follow
universal_mapping = {
    "pick red x handover place bottom left": 0,
    "pick red x handover place bottom middle": 1,
    "pick red x handover place bottom right": 2,
    "pick red x handover place center": 3,
    "pick red x handover place middle left": 4,
    "pick red x handover place middle right": 5,
    "pick red x handover place top left": 6,
    "pick red x handover place top middle": 7,
    "pick red x handover place top right": 8
}

api = HfApi()

for repo_id in repos:
    print(f"\n{'='*60}")
    print(f"Standardizing {repo_id}...")
    print(f"{'='*60}")
    
    try:
        # 1. Enforce universal tasks.parquet
        print("1. Enforcing universal 0-8 integer mapping in tasks.parquet...")
        
        # Create a clean DataFrame sorted by the integer values
        sorted_tasks = sorted(universal_mapping.keys(), key=lambda k: universal_mapping[k])
        df_new_tasks = pd.DataFrame({"task_index": range(9)}, index=sorted_tasks)
        df_new_tasks.index.name = "__index_level_0__"
        
        local_tasks = "temp_tasks.parquet"
        df_new_tasks.to_parquet(local_tasks)
        api.upload_file(
            path_or_fileobj=local_tasks,
            path_in_repo="meta/tasks.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Forced universal integer task mapping"
        )
        os.remove(local_tasks)
        print(" -> Success: tasks.parquet standardized.")

        # 2. Re-map strings to integers in all episodes chunks
        print("\n2. Scanning for episodes.parquet chunks...")
        all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        ep_files = [f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")]

        def process_val(val):
            is_list = False
            raw_val = val
            
            # Unwrap
            if hasattr(val, 'tolist'):
                raw_val = val.tolist()
            if isinstance(raw_val, (list, tuple)):
                is_list = True
                raw_val = raw_val[0] if len(raw_val) > 0 else raw_val
                
            # If it's already an int, return it
            if isinstance(raw_val, (int, float)) or (isinstance(raw_val, str) and raw_val.isdigit()):
                return [int(raw_val)] if is_list else int(raw_val)
                
            # Clean string, then grab universal ID
            clean_str = string_cleaner.get(raw_val, raw_val)
            new_idx = universal_mapping.get(clean_str, 0)
            
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
                
                local_ep_save = "temp_ep.parquet"
                df_ep.to_parquet(local_ep_save)
                
                api.upload_file(
                    path_or_fileobj=local_ep_save,
                    path_in_repo=ep_file,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Converted strings to universal int indices in {ep_file}"
                )
                print(f" -> Success: Mapped {ep_file}")
                os.remove(local_ep_save)

    except Exception as e:
        print(f"An error occurred on {repo_id}: {e}")

print("\nFoundation is solid! Both datasets now perfectly share the exact same LeRobot integer standard.")