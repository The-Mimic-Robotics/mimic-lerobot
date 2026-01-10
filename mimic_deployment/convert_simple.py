#!/usr/bin/env python3
"""
Simple converter: Use LeRobot's official v21->v30 converter, then zero-pad actions.
"""
import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo-id", required=True, help="Old dataset repo ID")
    parser.add_argument("--output-repo-id", required=True, help="New dataset repo ID")
    args = parser.parse_args()
    
    logger.info(f"Step 1: Converting {args.input_repo_id} using LeRobot's official converter...")
    
    # Use LeRobot's official v21->v30 converter
    result = subprocess.run([
        "python", "-m", "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id={args.input_repo_id}",
        "--local-only",
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        return 1
    
    logger.info("Official conversion completed")
    
    # Now modify the converted dataset locally
    cache_dir = Path.home() / ".cache/huggingface/lerobot" / args.input_repo_id
    logger.info(f"Step 2: Zero-padding actions in {cache_dir}...")
    
    # Modify all data parquet files
    data_dir = cache_dir / "data"
    for parquet_file in data_dir.rglob("*.parquet"):
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Zero-pad action column
        if "action" in df.columns:
            old_actions = np.stack(df["action"].values)
            zeros = np.zeros((len(old_actions), 3))
            new_actions = np.concatenate([old_actions, zeros], axis=1)
            df["action"] = list(new_actions)
        
        # Zero-pad observation.state column
        if "observation.state" in df.columns:
            old_state = np.stack(df["observation.state"].values)
            zeros = np.zeros((len(old_state), 3))
            new_state = np.concatenate([old_state, zeros], axis=1)
            df["observation.state"] = list(new_state)
        
        # Write back
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_file)
    
    logger.info("Step 3: Pushing to new repo...")
    api = HfApi()
    api.create_repo(args.output_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=cache_dir,
        repo_id=args.output_repo_id,
        repo_type="dataset",
    )
    
    logger.info(f"✅ Done: {args.output_repo_id}")

if __name__ == "__main__":
    main()
