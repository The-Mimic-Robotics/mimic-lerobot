#!/usr/bin/env python3
"""
Script to tag the mimic_mobile_bimanual_drift_v2 dataset with the required version tag.
Run this with your Hugging Face credentials that have write access to the Mimic-Robotics org.
"""

from huggingface_hub import HfApi

def tag_dataset():
    api = HfApi()
    
    repo_id = "Mimic-Robotics/mimic_mobile_bimanual_drift_v2"
    tag = "v3.0"
    
    print(f"Creating tag '{tag}' for dataset: {repo_id}")
    
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=tag,
            repo_type="dataset",
            tag_message="LeRobot v3.0 dataset format"
        )
        print(f"✓ Successfully created tag: {tag}")
        print(f"\nYou can now run the training script!")
        
    except Exception as e:
        print(f"❌ Error creating tag: {e}")
        print(f"\nPlease make sure:")
        print(f"  1. You are logged in: huggingface-cli login")
        print(f"  2. Your token has write access to {repo_id}")
        print(f"\nOr run this command manually:")
        print(f"```python")
        print(f"from huggingface_hub import HfApi")
        print(f"api = HfApi()")
        print(f'api.create_tag("{repo_id}", tag="{tag}", repo_type="dataset")')
        print(f"```")

if __name__ == "__main__":
    tag_dataset()
