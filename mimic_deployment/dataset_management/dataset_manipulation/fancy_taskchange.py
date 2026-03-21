import os
import random
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, list_repo_files, duplicate_repo

# --- Configuration ---
source_repo_id = "Mimic-Robotics/mimic_ttt_redx_ALLBAL_15hz"
target_repo_id = "Mimic-Robotics/mimic_ttt_redx_ALLBAL_15hz_fancy"

api = HfApi()

# 1. Your map to clean up any messy strings in the original dataset
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

# 2. The Natural Language Augmentations
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
    duplicate_repo(from_id=source_repo_id, to_id=target_repo_id, repo_type="dataset")
    print(" -> Clone successful!\n")
except Exception as e:
    print(f" -> Repo might already exist or permission denied: {e}\n")

try:
    # 3. CRITICAL FIX: Read the original tasks.parquet to get the absolute ground-truth mapping
    print("Reading ground-truth integer mapping from original dataset...")
    original_tasks_file = hf_hub_download(repo_id=source_repo_id, repo_type="dataset", filename="meta/tasks.parquet")
    df_orig_tasks = pd.read_parquet(original_tasks_file)
    
    old_int_to_string = {}
    if "task_index" in df_orig_tasks.columns:
        old_int_to_string = {row['task_index']: str(idx_str) for idx_str, row in df_orig_tasks.iterrows()}
    else:
        old_int_to_string = {i: str(task) for i, task in enumerate(df_orig_tasks.index)}
        
    print(f" -> Found {len(old_int_to_string)} original mappings.\n")

    # 4. Create the new tasks mapping
    all_fancy_tasks = sorted([sentence for sentences in task_variations.values() for sentence in sentences])
    df_new_tasks = pd.DataFrame({"task_index": range(len(all_fancy_tasks))}, index=all_fancy_tasks)
    df_new_tasks.index.name = "__index_level_0__"
    
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
    
    fancy_string_to_new_idx = {sentence: idx for idx, sentence in enumerate(all_fancy_tasks)}

    # 5. Modify the episodes.parquet chunks
    print("Scanning for episodes.parquet chunks to randomize instructions...")
    all_files = list_repo_files(repo_id=target_repo_id, repo_type="dataset")
    ep_files = [f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")]

    def process_val(val):
        is_list_or_array = False
        length = 1
        raw_val = val
        
        # Unwrap lists/arrays safely to find the length of the video sequence
        if hasattr(val, 'tolist'):
            raw_val = val.tolist()
        if isinstance(raw_val, (list, tuple)):
            is_list_or_array = True
            length = len(raw_val)
            raw_val = raw_val[0] if length > 0 else raw_val
            
        # Get the original string using the ground-truth mapping
        original_string = None
        if isinstance(raw_val, (int, float)) or (isinstance(raw_val, str) and str(raw_val).isdigit()):
            original_string = old_int_to_string.get(int(raw_val))
        else:
            original_string = str(raw_val)
            
        # Clean it using your task_map
        clean_string = task_map.get(original_string, original_string)

        # Pick ONE random variation for this specific episode
        if clean_string in task_variations:
            chosen_fancy = random.choice(task_variations[clean_string])
        else:
            chosen_fancy = random.choice(task_variations["pick red x handover place center"])
            
        new_idx = fancy_string_to_new_idx[chosen_fancy]
        
        # CRITICAL FIX: Return an array of the exact same length if it was an array
        return [new_idx] * length if is_list_or_array else new_idx

    for ep_file in ep_files:
        local_ep = hf_hub_download(repo_id=target_repo_id, repo_type="dataset", filename=ep_file)
        df_ep = pd.read_parquet(local_ep)
        
        task_col = None
        for col in ['tasks', 'task_index', 'task']:
            if col in df_ep.columns:
                task_col = col
                break
                
        if task_col:
            df_ep[task_col] = df_ep[task_col].apply(process_val)
            
            local_ep_save = "fixed_ep_fancy.parquet"
            df_ep.to_parquet(local_ep_save)
            
            api.upload_file(
                path_or_fileobj=local_ep_save,
                path_in_repo=ep_file,
                repo_id=target_repo_id,
                repo_type="dataset",
                commit_message=f"Applied randomized natural language instructions to {ep_file}"
            )
            print(f" -> Success: Randomized tasks in {ep_file}")
            os.remove(local_ep_save)

    print("\nAll done! Your _fancy dataset is correctly mapped and randomized.")

except Exception as e:
    print(f"An error occurred: {e}")