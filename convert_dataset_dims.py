#!/usr/bin/env python
"""
Script to convert LeRobot datasets from 12D action/state space to 15D.
This is done by appending 3 zeros (representing base) to the action and observation vectors.
"""

import argparse
from pathlib import Path
import logging
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import modify_features

logging.basicConfig(level=logging.INFO)

def pad_vector(row, ep_idx, frame_idx, key, pad_width=3):
    """Pads a vector with zeros."""
    vec = row[key]
    if isinstance(vec, torch.Tensor):
        vec = vec.numpy()
    
    # Check if already 15D
    if vec.shape[0] == 15:
        return vec
        
    if vec.shape[0] != 12:
         # Log warning only once per key per run to avoid spam? 
         # For now just return as is if not 12, assuming it might be something else 
         # but strict conversion requires 12->15.
         # Actually let's error or warn.
         pass

    # Append zeros
    new_vec = np.concatenate([vec, np.zeros(pad_width, dtype=vec.dtype)])
    return new_vec

def convert_dataset(repo_id, output_repo_id=None, push_to_hub=False):
    if output_repo_id is None:
        output_repo_id = f"{repo_id}_15d"

    logging.info(f"Converting {repo_id} to {output_repo_id} (12D -> 15D)")
    
    # Load dataset
    ds = LeRobotDataset(repo_id)
    
    # Check current dimensions
    current_action_dim = ds.meta.features["action"]["shape"][0]
    logging.info(f"Current action dimension: {current_action_dim}")
    
    if current_action_dim == 15:
        logging.info("Dataset is already 15D. Skipping.")
        return

    if current_action_dim != 12:
        logging.warning(f"Dataset has {current_action_dim} dimensions, expected 12. Proceeding with padding anyway?")
        # We'll pad difference
        pad = 15 - current_action_dim
        if pad < 0:
             logging.error("Dataset has more than 15 dimensions. Skipping.")
             return
    else:
        pad = 3

    # Define new features
    # usage: modify_features allows overwriting if we use the same name? 
    # Waiting for confirmation on modify_features overwriting. 
    # dataset_tools.py checks logic: if key in features, raise ValueError.
    # So we MUST remove then add.
    
    # But as analyzed, if we remove 'action', the row passed to add_features callable won't have it.
    # So we need a custom approach: 
    # 1. Add 'action_15d' (reading 'action')
    # 2. Remove 'action'
    # 3. Add 'action' (reading 'action_15d')
    # 4. Remove 'action_15d'
    # This is verbose but safe using existing tools.
    
    # Step 1: Add intermediate 15d features
    logging.info("Step 1: Creating temporary 15D features...")
    
    add_temp_features = {}
    
    # Action
    add_temp_features["action_temp"] = (
         lambda row, ep, frame: pad_vector(row, ep, frame, "action", pad),
         {"dtype": "float32", "shape": (15,), "names": None} # names can be updated later?
    )
    
    # Observation State
    if "observation.state" in ds.meta.features:
        add_temp_features["observation.state_temp"] = (
             lambda row, ep, frame: pad_vector(row, ep, frame, "observation.state", pad),
             {"dtype": "float32", "shape": (15,), "names": None}
        )

    import shutil

    step1_repo_id = f"{output_repo_id}_step1"
    if (ds.root.parent / step1_repo_id.split('/')[-1]).exists():
        shutil.rmtree(ds.root.parent / step1_repo_id.split('/')[-1])

    ds_step1 = modify_features(
        dataset=ds,
        add_features=add_temp_features,
        repo_id=step1_repo_id,
    )

    # Step 2: Remove old features
    logging.info("Step 2: Removing old 12D features...")
    features_to_remove = ["action"]
    if "observation.state" in ds.meta.features:
        features_to_remove.append("observation.state")
        
    step2_repo_id = f"{output_repo_id}_step2"
    if (ds.root.parent / step2_repo_id.split('/')[-1]).exists():
        shutil.rmtree(ds.root.parent / step2_repo_id.split('/')[-1])

    ds_step2 = modify_features(
        dataset=ds_step1,
        remove_features=features_to_remove,
        repo_id=step2_repo_id,
    )
    
    # Step 3: Rename temp to proper names (Add new reading temp, Remove temp)
    logging.info("Step 3: Renaming features back...")
    
    add_final_features = {}
    add_final_features["action"] = (
        lambda row, ep, frame: row["action_temp"],
        {"dtype": "float32", "shape": (15,), "names": None}
    )
    if "observation.state_temp" in ds_step2.meta.features:
        add_final_features["observation.state"] = (
             lambda row, ep, frame: row["observation.state_temp"],
             {"dtype": "float32", "shape": (15,), "names": None}
        )
        
    step3_repo_id = f"{output_repo_id}_step3"
    if (ds.root.parent / step3_repo_id.split('/')[-1]).exists():
        shutil.rmtree(ds.root.parent / step3_repo_id.split('/')[-1])

    ds_step3 = modify_features(
        dataset=ds_step2,
        add_features=add_final_features,
        repo_id=step3_repo_id
    )
    
    # Step 4: Remove temp features
    logging.info("Step 4: Cleanup...")
    remove_temp = ["action_temp"]
    if "observation.state_temp" in ds_step3.meta.features:
        remove_temp.append("observation.state_temp")
    
    # Final output
    # Ensure it doesn't exist
    if (ds.root.parent / output_repo_id.split('/')[-1]).exists():
        shutil.rmtree(ds.root.parent / output_repo_id.split('/')[-1])

    ds_final = modify_features(
        dataset=ds_step3,
        remove_features=remove_temp,
        repo_id=output_repo_id
    )

    if push_to_hub:
        logging.info("Pushing to Hub...")
        ds_final.push_to_hub()

    logging.info("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Source dataset repo id")
    parser.add_argument("--output-repo-id", type=str, help="Output dataset repo id")
    parser.add_argument("--push", action="store_true", help="Push to hub")
    args = parser.parse_args()
    
    convert_dataset(args.repo_id, args.output_repo_id, args.push)
