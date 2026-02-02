#!/usr/bin/env python3
"""
Test the converted dataset using LeRobot's low-level utilities.
This verifies the data format without requiring HuggingFace Hub access.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.utils import load_json, load_episodes, load_nested_dataset
import pyarrow.parquet as pq

def main():
    dataset_path = Path("/tmp/dataset_conversion_cvqdnpvv/Mimic-Robotics/bimanual_blue_block_handover_1")
    
    print("="*70)
    print("VERIFYING CONVERTED DATASET STRUCTURE")
    print("="*70)
    print(f"\nDataset path: {dataset_path}\n")
    
    try:
        # Test 1: Load info.json
        print("Test 1: Loading metadata...")
        info_path = dataset_path / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        
        print(f"  ✅ Codebase version: {info['codebase_version']}")
        print(f"  ✅ Robot type: {info['robot_type']}")
        print(f"  ✅ Total episodes: {info['total_episodes']}")
        print(f"  ✅ Action dimensions: {info['features']['action']['shape'][0]}")
        print(f"  ✅ Camera keys: {[k for k in info['features'] if 'images' in k]}")
        
        # Verify critical fields
        assert info['codebase_version'] == 'v3.0', f"Expected v3.0, got {info['codebase_version']}"
        assert info['robot_type'] == 'mimic_follower', f"Expected mimic_follower, got {info['robot_type']}"
        assert info['features']['action']['shape'][0] == 15, f"Expected 15D actions"
        assert info['features']['observation.state']['shape'][0] == 15, f"Expected 15D observations"
        
        # Test 2: Load a data parquet file
        print(f"\nTest 2: Loading data files...")
        data_file = dataset_path / "data" / "chunk-000" / "episode_000000.parquet"
        table = pq.read_table(data_file)
        df = table.to_pandas()
        
        print(f"  ✅ Episode 0 has {len(df)} frames")
        print(f"  ✅ Action shape: {df['action'].iloc[0].shape}")
        print(f"  ✅ Observation state shape: {df['observation.state'].iloc[0].shape}")
        
        # Verify dimensions
        assert df['action'].iloc[0].shape == (15,), f"Expected 15D actions in data"
        assert df['observation.state'].iloc[0].shape == (15,), f"Expected 15D observations in data"
        
        # Test 3: Check video directories
        print(f"\nTest 3: Checking video structure...")
        video_dir = dataset_path / "videos"
        cameras = list(video_dir.iterdir())
        camera_names = [c.name for c in cameras if c.is_dir()]
        print(f"  ✅ Found {len(camera_names)} cameras: {camera_names}")
        
        expected_cameras = [
            'observation.images.front',
            'observation.images.left_wrist',
            'observation.images.right_wrist',
            'observation.images.top'
        ]
        
        for cam in expected_cameras:
            cam_path = video_dir / cam / "chunk-000"
            if not cam_path.exists():
                print(f"  ❌ Missing camera: {cam}")
                return False
            
            videos = list(cam_path.glob("*.mp4"))
            print(f"  ✅ {cam}: {len(videos)} videos")
        
        # Test 4: Verify action names
        print(f"\nTest 4: Verifying action/observation names...")
        action_names = info['features']['action']['names']
        print(f"  Action names: {action_names[-3:]}")  # Show last 3 (base dims)
        
        assert 'base_vx' in action_names, "Missing base_vx"
        assert 'base_vy' in action_names, "Missing base_vy"
        assert 'base_omega' in action_names, "Missing base_omega"
        print(f"  ✅ Base dimensions present in actions")
        
        obs_names = info['features']['observation.state']['names']
        print(f"  Observation names: {obs_names[-3:]}")  # Show last 3 (base dims)
        
        assert 'base_x' in obs_names, "Missing base_x"
        assert 'base_y' in obs_names, "Missing base_y"
        assert 'base_theta' in obs_names, "Missing base_theta"
        print(f"  ✅ Base dimensions present in observations")
        
        print("\n" + "="*70)
        print("✅ ALL VALIDATIONS PASSED")
        print("✅ DATASET IS CORRECTLY FORMATTED FOR MIMIC_FOLLOWER ROBOTS")
        print("="*70)
        print(f"\n🎉 Dataset ready at: {dataset_path}")
        print(f"\nYou can now:")
        print(f"  1. Push this dataset to HuggingFace")
        print(f"  2. Use it for training with LeRobot")
        print(f"  3. Convert all 21 datasets using the same pipeline")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
