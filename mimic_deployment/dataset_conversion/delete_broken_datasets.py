#!/usr/bin/env python3
"""
Delete the 20 broken mobile_bimanual_blue_block_handover_* datasets from HuggingFace.
These datasets have wrong camera names and need to be replaced with properly converted versions.
"""

from huggingface_hub import HfApi
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of broken datasets to delete
BROKEN_DATASETS = [
    "mobile_bimanual_blue_block_handover_1",
    "mobile_bimanual_blue_block_handover_2",
    "mobile_bimanual_blue_block_handover_3",
    "mobile_bimanual_blue_block_handover_4",
    "mobile_bimanual_blue_block_handover_5",
    "mobile_bimanual_blue_block_handover_6",
    "mobile_bimanual_blue_block_handover_7",
    "mobile_bimanual_blue_block_handover_14",
    "mobile_bimanual_blue_block_handover_15",
    "mobile_bimanual_blue_block_handover_16",
    "mobile_bimanual_blue_block_handover_17",
    "mobile_bimanual_blue_block_handover_18",
    "mobile_bimanual_blue_block_handover_19",
    "mobile_bimanual_blue_block_handover_20",
    "mobile_bimanual_blue_block_handover_21",
    "mobile_bimanual_blue_block_handover_22",
    "mobile_bimanual_blue_block_handover_23",
    "mobile_bimanual_blue_block_handover_24",
    "mobile_bimanual_blue_block_handover_25",
    "mobile_bimanual_blue_block_handover_26",
]

ORG = "Mimic-Robotics"


def main():
    api = HfApi()
    
    print("\n" + "="*70)
    print("DELETING BROKEN DATASETS FROM HUGGINGFACE")
    print("="*70)
    print(f"\nTotal datasets to delete: {len(BROKEN_DATASETS)}")
    print("\nThese datasets have:")
    print("  - Wrong camera names (wrist_right instead of right_wrist)")
    print("  - Missing front camera")
    print("  - Incompatible with latest format")
    print("\n" + "="*70)
    
    response = input("\nAre you sure you want to delete these datasets? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        return
    
    print("\nDeleting datasets...")
    
    success_count = 0
    failed_count = 0
    
    for dataset_name in BROKEN_DATASETS:
        repo_id = f"{ORG}/{dataset_name}"
        try:
            logger.info(f"Deleting {repo_id}...")
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
            logger.info(f"  Successfully deleted {repo_id}")
            success_count += 1
        except Exception as e:
            logger.error(f"  Failed to delete {repo_id}: {e}")
            failed_count += 1
    
    print("\n" + "="*70)
    print("DELETION COMPLETE")
    print("="*70)
    print(f"Successfully deleted: {success_count}")
    print(f"Failed: {failed_count}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
