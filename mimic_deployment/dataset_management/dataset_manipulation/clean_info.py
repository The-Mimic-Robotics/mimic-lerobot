import json
import os
from huggingface_hub import hf_hub_download, HfApi

repos = [
    "Mimic-Robotics/mimic_ttt_redx_15hz_x2",
    "Mimic-Robotics/mimic_ttt_redBalanced_15hz"
]

# The universal 0-8 task list
clean_tasks = [
    "pick red x handover place bottom left",
    "pick red x handover place bottom middle",
    "pick red x handover place bottom right",
    "pick red x handover place center",
    "pick red x handover place middle left",
    "pick red x handover place middle right",
    "pick red x handover place top left",
    "pick red x handover place top middle",
    "pick red x handover place top right"
]

api = HfApi()

for repo_id in repos:
    print(f"Syncing info.json for {repo_id}...")
    try:
        # Download
        info_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/info.json")
        
        with open(info_file, "r") as f:
            info_data = json.load(f)
            
        # Overwrite tasks dictionary
        info_data['tasks'] = clean_tasks
        
        # Save locally
        local_info = f"temp_info_{repo_id.split('/')[-1]}.json"
        with open(local_info, "w") as f:
            json.dump(info_data, f, indent=4)
            
        # Push
        api.upload_file(
            path_or_fileobj=local_info,
            path_in_repo="meta/info.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Forced universal task list into info.json"
        )
        os.remove(local_info)
        print(f" -> Success! {repo_id} is ready for merging.\n")
        
    except Exception as e:
        print(f"Error on {repo_id}: {e}")