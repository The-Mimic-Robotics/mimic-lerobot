import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
import os

# Your explicit list of remote repositories
repos = [
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BL_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BL_v3",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BM_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BM_v3",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BR_v1",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BR_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BR_v3",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TL_v1",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TL_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TL_v3",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_ML_v1",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_ML_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_ML_v3",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_center_v1",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_center_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_MR_v1",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_MR_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TM_v1",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TM_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TR_v1",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TR_v2",
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TR_v3"
]

# Map the acronyms in your repo names to the actual text you want
position_map = {
    "_TL_": "top left",
    "_TM_": "top middle",
    "_TR_": "top right",
    "_ML_": "middle left",
    "_center_": "center",
    "_MR_": "middle right",
    "_BL_": "bottom left",
    "_BM_": "bottom middle",
    "_BR_": "bottom right"
}

api = HfApi()
local_save_path = "temp_tasks.parquet"

print(f"Starting batch update for {len(repos)} datasets...\n")

for repo_id in repos:
    print(f"Processing: {repo_id}")
    
    # 1. Determine the correct task name based on the repo name
    target_position = None
    for key, position_text in position_map.items():
        if key in repo_id:
            target_position = position_text
            break
            
    if not target_position:
        print(f"  -> Skipped: Could not infer position from repo name '{repo_id}'\n")
        continue

    new_task_name = f"pick red x handover place {target_position}"
    
    try:
        # 2. Download the tasks.parquet for this specific repo
        file_path = hf_hub_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            filename="meta/tasks.parquet"
        )

        # 3. Load, update index, and save locally
        df = pd.read_parquet(file_path)
        
        if len(df) == 1:
            df.index = pd.Index([new_task_name], name=df.index.name)
            df.to_parquet(local_save_path)

            # 4. Push directly back to the dataset repo
            api.upload_file(
                path_or_fileobj=local_save_path,
                path_in_repo="meta/tasks.parquet",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Batch update task name to: {new_task_name}"
            )
            print(f"  -> Success: Set to '{new_task_name}'")
        else:
            print(f"  -> Skipped: Found {len(df)} tasks instead of 1.")

    except Exception as e:
        print(f"  -> Failed: {e}")
    
    print("-" * 50)

# Clean up
if os.path.exists(local_save_path):
    os.remove(local_save_path)

print("\nBatch update complete!")