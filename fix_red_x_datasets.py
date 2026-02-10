#!/usr/bin/env python3
"""
Script to fix red_x_handover_place datasets:
1. Tag them with the latest codebase version (v3.0)
2. Update task descriptions to "pick red x handover place center"
"""
import pandas as pd
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, upload_file, HfApi
from huggingface_hub.utils import RepositoryNotFoundError

# --- CONFIGURATION ---
BASE_REPO_NAME = "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm"
FILE_PATH_IN_REPO = "meta/tasks.parquet"
NEW_TASK_DESCRIPTION = "pick red x handover place center"
CODEBASE_VERSION = "v3.0"

START_VERSION = 2  # Start from v2 (v1 doesn't exist)
END_VERSION = 16

print("=" * 80)
print("RED X HANDOVER PLACE DATASET FIX SCRIPT")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Base repo: {BASE_REPO_NAME}")
print(f"  Versions: v{START_VERSION} to v{END_VERSION}")
print(f"  New task description: '{NEW_TASK_DESCRIPTION}'")
print(f"  Codebase version tag: '{CODEBASE_VERSION}'")
print(f"\n" + "=" * 80 + "\n")

api = HfApi()
success_count = 0
skip_count = 0
error_count = 0

for i in range(START_VERSION, END_VERSION + 1):
    repo_id = f"{BASE_REPO_NAME}_v{i}"
    
    print(f"[{i:2d}/16] Processing {repo_id}...")
    
    try:
        # STEP 1: Tag the dataset with codebase version
        print(f"       -> Tagging with {CODEBASE_VERSION}...", end=" ", flush=True)
        try:
            api.create_tag(
                repo_id=repo_id,
                tag=CODEBASE_VERSION,
                repo_type="dataset",
                exist_ok=True,
            )
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})")
            # Continue anyway, as tagging might not be critical
        
        # STEP 2: Download and check the parquet file
        print(f"       -> Downloading metadata...", end=" ", flush=True)
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=FILE_PATH_IN_REPO
        )
        print("✓")
        
        # STEP 3: Read and check current task description
        print(f"       -> Reading task description...", end=" ", flush=True)
        df = pd.read_parquet(local_path)
        # Task description is stored in the index (row name)
        current_text = df.index[0]
        print("✓")
        
        # STEP 4: Check if update is needed
        if current_text == NEW_TASK_DESCRIPTION:
            print(f"       -> Task already correct: '{current_text}'")
            skip_count += 1
            continue
        
        print(f"       -> Current task: '{current_text}'")
        print(f"       -> Updating to: '{NEW_TASK_DESCRIPTION}'...", end=" ", flush=True)
        
        # STEP 5: Update the task description by renaming the index
        df.index = [NEW_TASK_DESCRIPTION]
        
        # Save to temporary file
        temp_file = f"temp_tasks_v{i}.parquet"
        df.to_parquet(temp_file)
        
        # STEP 6: Upload back to Hugging Face
        upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=FILE_PATH_IN_REPO,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update task description to: {NEW_TASK_DESCRIPTION}"
        )
        
        # Cleanup
        os.remove(temp_file)
        print("✓")
        
        print(f"       -> [SUCCESS] v{i} tagged and updated\n")
        success_count += 1

    except RepositoryNotFoundError:
        print(f"✗ (Repository not found)")
        print(f"       -> [SKIPPED] v{i} does not exist\n")
        skip_count += 1
    except Exception as e:
        print(f"✗")
        print(f"       -> [ERROR] Could not process v{i}: {str(e)}\n")
        error_count += 1

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Successfully updated: {success_count}")
print(f"⊘ Skipped (already correct): {skip_count}")
print(f"✗ Errors: {error_count}")
print("=" * 80)

if error_count == 0:
    print("\n✓ All datasets have been successfully tagged and updated!")
    print("  You can now train with:")
    print("  ./mimic_deployment/training_scripts/train_manager.sh --policy act --dataset-group red_x_handover_and_place_center")
else:
    print(f"\n✗ {error_count} dataset(s) failed. Please review the errors above.")

print("\nNote: The LeRobot Visualizer may take 5-10 minutes to refresh.")
