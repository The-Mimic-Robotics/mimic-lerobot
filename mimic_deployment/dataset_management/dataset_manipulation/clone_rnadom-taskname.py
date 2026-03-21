import os
import random
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, list_repo_files, duplicate_repo

# --- Configuration ---
source_repo_id = "Mimic-Robotics/mimic_ttt_redx_ALLBAL_15hz"
target_repo_id = "Mimic-Robotics/mimic_ttt_redx_ALLBAL_15hz_fancy"

api = HfApi()

# 1. Define the Natural Language Augmentations
# We map your 9 base tasks to a list of rich, varied instructions.
task_variations = {
    "pick red x handover place top left": [
        "Pick up the red X, pass it to the other arm, and place it in the top left corner of the board.",
        "Grab a red cross, perform a handover, and set it down on the upper left.",
        "Play a red X in the top left square.",
        "Put the red game piece in the top left position.",
    ],
    "pick red x handover place top middle": [
        "Pick up the red X, pass it to the other arm, and place it in the top middle of the board.",
        "Grab a red cross, perform a handover, and set it down on the upper middle square.",
        "Play a red X in the top center position.",
        "Put the red game piece in the top middle square.",
    ],
    "pick red x handover place top right": [
        "Pick up the red X, pass it to the other arm, and place it in the top right corner of the board.",
        "Grab a red cross, perform a handover, and set it down on the upper right.",
        "Play a red X in the top right square.",
        "Put the red game piece in the top right position.",
    ],
    "pick red x handover place middle left": [
        "Pick up the red X, pass it to the other arm, and place it in the middle left square.",
        "Grab a red cross, perform a handover, and set it down on the center left.",
        "Play a red X on the middle left of the grid.",
        "Put the red game piece in the left middle position.",
    ],
    "pick red x handover place center": [
        "Pick up the red X, pass it to the other arm, and place it directly in the center of the board.",
        "Grab a red cross, perform a handover, and set it down in the middle square.",
        "Play a red X in the dead center of the grid.",
        "Put the red game piece in the central position.",
    ],
    "pick red x handover place middle right": [
        "Pick up the red X, pass it to the other arm, and place it in the middle right square.",
        "Grab a red cross, perform a handover, and set it down on the center right.",
        "Play a red X on the middle right of the grid.",
        "Put the red game piece in the right middle position.",
    ],
    "pick red x handover place bottom left": [
        "Pick up the red X, pass it to the other arm, and place it in the bottom left corner of the board.",
        "Grab a red cross, perform a handover, and set it down on the lower left.",
        "Play a red X in the bottom left square.",
        "Put the red game piece in the bottom left position.",
    ],
    "pick red x handover place bottom middle": [
        "Pick up the red X, pass it to the other arm, and place it in the bottom middle of the board.",
        "Grab a red cross, perform a handover, and set it down on the lower middle square.",
        "Play a red X in the bottom center position.",
        "Put the red game piece in the bottom middle square.",
    ],
    "pick red x handover place bottom right": [
        "Pick up the red X, pass it to the other arm, and place it in the bottom right corner of the board.",
        "Grab a red cross, perform a handover, and set it down on the lower right.",
        "Play a red X in the bottom right square.",
        "Put the red game piece in the bottom right position.",
    ]
}

print(f"Cloning {source_repo_id} to {target_repo_id}...")
try:
    # This clones the entire dataset (including heavy videos) server-side instantly.
    duplicate_repo(from_id=source_repo_id, to_id=target_repo_id, repo_type="dataset")
    print(" -> Clone successful!\n")
except Exception as e:
    print(f" -> Repo might already exist or permission denied: {e}\n")

print("Building new augmented tasks.parquet...")

try:
    # 2. Reconstruct the old mapping to know what the current integers mean
    # (Based on your previous script, the old integers were assigned alphabetically)
    old_base_tasks = sorted(list(task_variations.keys()))
    old_idx_to_base_string = {idx: task for idx, task in enumerate(old_base_tasks)}
    
    # 3. Create the new tasks mapping
    # Flatten all variations into a single list
    all_fancy_tasks = sorted([sentence for sentences in task_variations.values() for sentence in sentences])
    
    # Create the new dataframe for tasks.parquet
    df_new_tasks = pd.DataFrame({"task_index": range(len(all_fancy_tasks))}, index=all_fancy_tasks)
    df_new_tasks.index.name = "__index_level_0__"
    
    # Save and upload the new tasks.parquet
    local_tasks = "fixed_tasks_fancy.parquet"
    df_new_tasks.to_parquet(local_tasks)
    api.upload_file(
        path_or_fileobj=local_tasks,
        path_in_repo="meta/tasks.parquet",
        repo_id=target_repo_id,
        repo_type="dataset",
        commit_message="Replaced static tasks with natural language augmentations"
    )
    os.remove(local_tasks)
    
    # Lookup dictionaries for the episode processing
    fancy_string_to_new_idx = {sentence: idx for idx, sentence in enumerate(all_fancy_tasks)}
    print(f" -> Generated {len(all_fancy_tasks)} unique natural language instructions.\n")

    # 4. Modify the episodes.parquet chunks
    print("Scanning for episodes.parquet chunks to randomize instructions...")
    all_files = list_repo_files(repo_id=target_repo_id, repo_type="dataset")
    ep_files = [f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")]

    def process_val(val):
        is_list = False
        raw_val = val
        
        # Unwrap lists/arrays safely
        if hasattr(val, 'tolist'):
            raw_val = val.tolist()
        if isinstance(raw_val, (list, tuple)):
            is_list = True
            raw_val = raw_val[0] if len(raw_val) > 0 else raw_val
            
        # Determine the base task string
        base_string = None
        if isinstance(raw_val, (int, float)) or (isinstance(raw_val, str) and raw_val.isdigit()):
            base_string = old_idx_to_base_string.get(int(raw_val))
        else:
            base_string = raw_val # In case it's still a string
            
        # Fallback if something goes wrong
        if base_string not in task_variations:
            print(f"Warning: Could not find base string for {raw_val}, defaulting to center.")
            base_string = "pick red x handover place center"

        # The Magic: Randomly sample one of the natural language variations
        chosen_fancy_string = random.choice(task_variations[base_string])
        new_idx = fancy_string_to_new_idx[chosen_fancy_string]
        
        return [new_idx] if is_list else new_idx

    for ep_file in ep_files:
        local_ep = hf_hub_download(repo_id=target_repo_id, repo_type="dataset", filename=ep_file)
        df_ep = pd.read_parquet(local_ep)
        
        task_col = None
        for col in ['tasks', 'task_index', 'task']:
            if col in df_ep.columns:
                task_col = col
                break
                
        if task_col:
            # Apply the randomized instruction mapping
            df_ep[task_col] = df_ep[task_col].apply(process_val)
            
            local_ep_save = "fixed_ep_fancy.parquet"
            df_ep.to_parquet(local_ep_save)
            
            # Push the modified metadata back to the Hub
            api.upload_file(
                path_or_fileobj=local_ep_save,
                path_in_repo=ep_file,
                repo_id=target_repo_id,
                repo_type="dataset",
                commit_message=f"Applied randomized natural language instructions to {ep_file}"
            )
            print(f" -> Success: Randomized tasks in {ep_file}")
            os.remove(local_ep_save)
        else:
            print(f" -> Warning: No task column found in {ep_file}")

    print("\nAll done! Your _fancy dataset is now populated with rich, randomized natural language instructions.")

except Exception as e:
    print(f"An error occurred: {e}")