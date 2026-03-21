import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

# Replace with your dataset repo ID
# repo_id = "Mimic-Robotics/mimic_ttt_redBalanced_15hz" 
# repo_id = "Mimic-Robotics/mimic_ttt_redx_30hz_x2_ALL" 
repo_id = "Mimic-Robotics/mimic_ttt_redx_ALLBAL_15hz"


print(f"Fetching metadata for {repo_id} to analyze tasks...\n")

try:
    # 1. Download and load tasks.parquet
    tasks_path = hf_hub_download(
        repo_id=repo_id, 
        repo_type="dataset", 
        filename="meta/tasks.parquet"
    )
    df_tasks = pd.read_parquet(tasks_path)
    
    # Map task_index -> task_name
    task_mapping = {}
    for task_name, row in df_tasks.iterrows():
        task_mapping[row['task_index']] = task_name

    # 2. Find all chunked episode files
    print("Scanning for episode metadata chunks...")
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    episode_files = [f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet")]
    
    if not episode_files:
        raise FileNotFoundError(f"Could not find any .parquet files in meta/episodes/ for {repo_id}")

    # 3. Download and combine all episode chunks
    df_list = []
    for ep_file in episode_files:
        ep_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=ep_file)
        df_list.append(pd.read_parquet(ep_path))
        
    df_episodes = pd.concat(df_list, ignore_index=True)

    # 4. Find the column that holds the task information
    task_col = None
    for col in ['tasks', 'task_index', 'task']:
        if col in df_episodes.columns:
            task_col = col
            break
            
    if not task_col:
        print(f"Error: Couldn't find task column. Found columns: {df_episodes.columns.tolist()}")
    else:
        # Safely extract integer
        def extract_task_idx(val):
            if hasattr(val, 'tolist'):
                val = val.tolist()
            if isinstance(val, (list, tuple)): 
                return val[0] if len(val) > 0 else -1
            return int(val)

        df_episodes['clean_task_idx'] = df_episodes[task_col].apply(extract_task_idx)
        
        # 5. Count the occurrences of each task index
        task_counts = df_episodes['clean_task_idx'].value_counts()
        
        # 6. Print the table
        print(f"\n{'Task Name':<55} | {'Episodes'}")
        print("-" * 70)
        
        total_episodes = 0
        for t_idx, count in task_counts.items():
            t_name = task_mapping.get(t_idx, f"Unknown Task (Index: {t_idx})")
            print(f"{t_name:<55} | {count}")
            total_episodes += count
            
        print("-" * 70)
        print(f"{'Total Episodes in Dataset':<55} | {total_episodes}")

except Exception as e:
    print(f"An error occurred: {e}")