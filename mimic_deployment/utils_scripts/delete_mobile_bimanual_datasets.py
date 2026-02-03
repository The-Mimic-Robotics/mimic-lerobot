#!/usr/bin/env python3
"""
Script to delete mobile_bimanual_blue_block_handover datasets from Hugging Face Hub.
Requires HF_TOKEN environment variable or HuggingFace CLI login.

Usage:
    python delete_mobile_bimanual_datasets.py
"""

from huggingface_hub import delete_repo, HfApi
import os

# Datasets to delete
DATASETS_TO_DELETE = [
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_1",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_2",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_3",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_4",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_5",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_6",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_7",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_14",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_15",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_16",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_17",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_18",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_19",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_20",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_21",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_22",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_23",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_24",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_25",
    "Mimic-Robotics/mobile_bimanual_blue_block_handover_26",
]

def main():
    # Get token from environment or use cached token
    token = os.getenv("HF_TOKEN")
    
    print(f"Will delete {len(DATASETS_TO_DELETE)} datasets from Hugging Face Hub")
    print("Datasets to delete:")
    for dataset in DATASETS_TO_DELETE:
        print(f"  - {dataset}")
    
    # Confirm before deletion
    response = input("\nAre you sure you want to delete these datasets? (yes/no): ")
    if response.lower() != "yes":
        print("Deletion cancelled.")
        return
    
    print("\nStarting deletion...")
    api = HfApi(token=token)
    
    success_count = 0
    failed_count = 0
    
    for dataset_id in DATASETS_TO_DELETE:
        try:
            print(f"Deleting {dataset_id}...", end=" ")
            delete_repo(
                repo_id=dataset_id,
                repo_type="dataset",
                token=token
            )
            print("✓ Success")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed_count += 1
    
    print(f"\nDeletion complete!")
    print(f"  Successfully deleted: {success_count}")
    print(f"  Failed: {failed_count}")

if __name__ == "__main__":
    main()
