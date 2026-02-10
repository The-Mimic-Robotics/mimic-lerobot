import pandas as pd
import os
from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils import RepositoryNotFoundError

# --- CONFIGURATION ---
# The base name of your repository (everything before the 'v')
BASE_REPO_NAME = "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm"

# The specific file path where LeRobot v3 stores task descriptions
FILE_PATH_IN_REPO = "meta/tasks/chunk-000/file-000.parquet"

# The new correct text you want for ALL these datasets
NEW_TASK_DESCRIPTION = "pick red x place center"

# Range of versions to fix (1 to 16)
START_VERSION = 1
END_VERSION = 16
# ---------------------

print(f"Starting Batch Fix for versions v{START_VERSION} to v{END_VERSION}...")
print(f"Target Description: '{NEW_TASK_DESCRIPTION}'\n")

for i in range(START_VERSION, END_VERSION + 1):
    # Construct the repo ID (e.g., ..._v1, ..._v2)
    # NOTE: If your versions are '_v01', change f"_v{i}" to f"_v{i:02d}"
    repo_id = f"{BASE_REPO_NAME}_v{i}"
    
    print(f"Checking {repo_id}...")

    try:
        # Step 1: Download the parquet file
        local_path = hf_hub_download(
            repo_id=repo_id, 
            repo_type="dataset",
            filename=FILE_PATH_IN_REPO
        )
        
        # Step 2: Read into Pandas
        df = pd.read_parquet(local_path)
        
        # Check current text to see if it actually needs changing
        current_text = df.iloc[0]['task']
        
        if current_text == NEW_TASK_DESCRIPTION:
            print(f"  -> [SKIP] Already fixed.")
            continue

        # Step 3: Modify the text
        # We update the 'task' column for the first row (index 0)
        df.at[0, 'task'] = NEW_TASK_DESCRIPTION
        
        # Save to a temporary local file
        temp_file_name = f"temp_tasks_v{i}.parquet"
        df.to_parquet(temp_file_name)
        
        # Step 4: Upload back to Hugging Face (Overwrites the old one)
        upload_file(
            path_or_fileobj=temp_file_name,
            path_in_repo=FILE_PATH_IN_REPO,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Fix task description to: {NEW_TASK_DESCRIPTION}"
        )
        
        # Clean up local file
        os.remove(temp_file_name)
        
        print(f"  -> [SUCCESS] Updated v{i}")

    except RepositoryNotFoundError:
        print(f"  -> [MISSING] Repo v{i} does not exist. Skipping.")
    except Exception as e:
        print(f"  -> [ERROR] Could not fix v{i}: {e}")

print("\nAll done! Please check the LeRobot Visualizer (it may take 5 mins to refresh).")
