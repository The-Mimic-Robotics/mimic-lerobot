#!/usr/bin/env python3
"""
Convert metadata from jsonl format to v3.0 parquet format.
"""

import json
import pandas as pd
from pathlib import Path
import shutil

def convert_metadata_to_v3(dataset_root: Path):
    """Convert episodes.jsonl to meta/episodes/chunk-000/*.parquet format."""
    meta_dir = dataset_root / "meta"
    
    # Read episodes.jsonl
    episodes_jsonl = meta_dir / "episodes.jsonl"
    if not episodes_jsonl.exists():
        print(f"ERROR: {episodes_jsonl} not found")
        return False
    
    with open(episodes_jsonl) as f:
        episodes = [json.loads(line) for line in f]
    
    print(f"Found {len(episodes)} episodes in episodes.jsonl")
    
    # Create meta/episodes/chunk-000 directory
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each episode to parquet
    for i, episode in enumerate(episodes):
        episode_df = pd.DataFrame([episode])
        episode_file = episodes_dir / f"file-{i:03d}.parquet"
        episode_df.to_parquet(episode_file, index=False)
        print(f"Created {episode_file.name}")
    
    # Convert episodes_stats.jsonl to stats.json (single file)
    episodes_stats_jsonl = meta_dir / "episodes_stats.jsonl"
    if episodes_stats_jsonl.exists():
        with open(episodes_stats_jsonl) as f:
            stats = [json.loads(line) for line in f]
        
        # Take the first entry (should only be one)
        if stats:
            stats_json = meta_dir / "stats.json"
            with open(stats_json, 'w') as f:
                json.dump(stats[0], f, indent=2)
            print(f"Created stats.json")
    
    # Remove old jsonl files
    if episodes_jsonl.exists():
        episodes_jsonl.unlink()
        print("Removed episodes.jsonl")
    
    if episodes_stats_jsonl.exists():
        episodes_stats_jsonl.unlink()
        print("Removed episodes_stats.jsonl")
    
    tasks_jsonl = meta_dir / "tasks.jsonl"
    if tasks_jsonl.exists():
        tasks_jsonl.unlink()
        print("Removed tasks.jsonl")
    
    return True

if __name__ == "__main__":
    dataset_root = Path("/tmp/dataset_conversion_cvqdnpvv/Mimic-Robotics/bimanual_blue_block_handover_1")
    
    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        exit(1)
    
    print(f"Converting metadata to v3.0 format...")
    print(f"Dataset root: {dataset_root}")
    print()
    
    success = convert_metadata_to_v3(dataset_root)
    
    if success:
        print("\n✅ Metadata conversion complete!")
    else:
        print("\n❌ Metadata conversion failed!")
        exit(1)
