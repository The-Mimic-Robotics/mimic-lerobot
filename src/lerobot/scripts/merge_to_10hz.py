

import logging
import json
import shutil
import numpy as np
import torch
from pathlib import Path

# --- IMPORTS ---
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from datasets import concatenate_datasets

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NEW_REPO_ID = "Mimic-Robotics/mimic_red_x_slow_10hz"
LOCAL_DIR = Path("data/merged_tictactoe_10hz")
PUSH_TO_HUB = True 
DOWNSAMPLE_TO_10HZ = True

SOURCE_REPOS = [
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v2",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v3",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v4",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v5",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v6",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v8",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topmiddle_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topMiddle_v3",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topMiddle_v4",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v3",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v4",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleRight_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleRight_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottom_middle_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottom_middle_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_center_slow_v1"
]

def make_serializable(obj):
    """Recursively convert numpy/torch arrays to python lists."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting merge of {len(SOURCE_REPOS)} datasets...")

    hf_datasets = []
    stats = None

    # 1. Load all datasets
    for i, repo in enumerate(SOURCE_REPOS):
        try:
            logging.info(f"Loading: {repo}")
            ds = LeRobotDataset(repo)
            
            # Grab stats from the first valid dataset
            if stats is None and hasattr(ds, 'meta') and hasattr(ds.meta, 'stats'):
                stats = ds.meta.stats
                logging.info("Found stats (normalization data) in this dataset.")

            hf_datasets.append(ds.hf_dataset)
        except Exception as e:
            logging.error(f"Failed to load {repo}: {e}")
            return

    # 2. Concatenate
    logging.info("Concatenating datasets...")
    merged_hf_ds = concatenate_datasets(hf_datasets)
    logging.info(f"Merged Total Frames (Raw 30Hz): {len(merged_hf_ds)}")

    # 3. Downsample to 10Hz
    if DOWNSAMPLE_TO_10HZ:
        logging.info("OPTIMIZING: Downsampling 30Hz -> 10Hz (Keeping every 3rd frame)...")
        indices_10hz = range(0, len(merged_hf_ds), 3)
        merged_hf_ds = merged_hf_ds.select(indices_10hz)
        logging.info(f"New Frame Count (Optimized 10Hz): {len(merged_hf_ds)}")

    # 4. Save to Disk
    logging.info("Removing runtime transforms for saving...")
    merged_hf_ds.reset_format() 
    
    logging.info(f"Saving merged dataset to {LOCAL_DIR}...")
    merged_hf_ds.save_to_disk(LOCAL_DIR)

    # 5. Save Stats (Sanitized)
    if stats:
        meta_dir = LOCAL_DIR / "meta"
        meta_dir.mkdir(exist_ok=True, parents=True)
        stats_path = meta_dir / "stats.json"
        
        logging.info(f"Saving stats to {stats_path}...")
        
        # Ensure it is a dict
        if hasattr(stats, "to_dict"):
            stats_dict = stats.to_dict()
        else:
            stats_dict = stats
            
        # Clean numpy arrays
        clean_stats = make_serializable(stats_dict)

        with open(stats_path, "w") as f:
            json.dump(clean_stats, f, indent=4)
    else:
        logging.warning("WARNING: No stats found! Training might fail without normalization stats.")

    # 6. Push to Hub
    if PUSH_TO_HUB:
        logging.info(f"Pushing to Hugging Face Hub: {NEW_REPO_ID}...")
        merged_hf_ds.push_to_hub(repo_id=NEW_REPO_ID)
        logging.info("Push complete!")

    logging.info("Done. Update your training script to point to this folder!")

if __name__ == "__main__":
    main()