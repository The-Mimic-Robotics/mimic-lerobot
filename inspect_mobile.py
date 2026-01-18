#!/usr/bin/env python
"""
Inspect a mobile_bimanual dataset to see if it is already 15D.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def inspect_mobile():
    repo_id = "Mimic-Robotics/mobile_bimanual_blue_block_handover_1"
    print(f"Inspecting {repo_id}...")
    
    try:
        ds = LeRobotDataset(repo_id)
        print(f"  Codebase version: {ds.meta.codebase_version}")
        print(f"  Action shape: {ds.meta.features['action']['shape']}")
        print(f"  State shape: {ds.meta.features['observation.state']['shape']}")
        
        is_15d = ds.meta.features['action']['shape'][0] == 15
        print(f"\nIS 15D? {'YES' if is_15d else 'NO'}")
        return is_15d
    except Exception as e:
        print(f"Failed to load: {e}")
        return False

if __name__ == "__main__":
    inspect_mobile()
