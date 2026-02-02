#!/usr/bin/env python3
"""
Verify converted dataset using LeRobot's official LeRobotDataset class.
This tests if the dataset can be loaded and used for training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

def main():
    dataset_root = Path("/tmp/dataset_conversion_cvqdnpvv/Mimic-Robotics")
    repo_id = "bimanual_blue_block_handover_1"
    
    print("="*70)
    print("VERIFYING CONVERTED DATASET USING LEROBOT's LeRobotDataset")
    print("="*70)
    print(f"\nDataset root: {dataset_root}")
    print(f"Repo ID: {repo_id}\n")
    
    try:
        # Load using LeRobot's official dataset class (local mode)
        print("Loading dataset with LeRobotDataset...")
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_root  # Local directory containing dataset folders
        )
        
        print("\n✅ Dataset loaded successfully!")
        print(f"\nDataset Info:")
        print(f"  - Robot type: {dataset.meta.robot_type}")
        print(f"  - Codebase version: {dataset.meta.codebase_version}")
        print(f"  - Total episodes: {dataset.meta.total_episodes}")
        print(f"  - Total frames: {dataset.meta.total_frames}")
        print(f"  - FPS: {dataset.meta.fps}")
        print(f"  - Cameras: {dataset.meta.camera_keys}")
        
        print(f"\nAction Space:")
        print(f"  - Shape: {dataset.meta.shapes['action']}")
        print(f"  - Dimension names: {dataset.meta.names['action']}")
        
        print(f"\nObservation State:")
        print(f"  - Shape: {dataset.meta.shapes['observation.state']}")
        print(f"  - Dimension names: {dataset.meta.names['observation.state']}")
        
        # Test data loading
        print(f"\n\nTesting data access...")
        sample = dataset[0]
        
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Action shape: {sample['action'].shape}")
        print(f"  Observation state shape: {sample['observation.state'].shape}")
        
        for cam_key in dataset.meta.camera_keys:
            print(f"  {cam_key} shape: {sample[cam_key].shape}")
        
        # Verify 15D actions
        assert sample['action'].shape == (15,), f"Expected 15D actions, got {sample['action'].shape}"
        assert sample['observation.state'].shape == (15,), f"Expected 15D obs, got {sample['observation.state'].shape}"
        
        # Verify camera images are valid
        for cam_key in dataset.meta.camera_keys:
            img = sample[cam_key]
            assert img.shape[0] == 3, f"Expected 3 channels for {cam_key}, got {img.shape[0]}"
            assert img.dtype == torch.float32, f"Expected float32 for {cam_key}, got {img.dtype}"
            assert img.min() >= 0.0 and img.max() <= 1.0, f"Image values out of range for {cam_key}"
        
        print(f"\n✅ All validations passed!")
        print(f"\n✅ Dataset is compatible with mimic_follower robots!")
        
        print("\n" + "="*70)
        print("VERIFICATION COMPLETE - DATASET READY FOR TRAINING")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
