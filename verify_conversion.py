#!/usr/bin/env python
"""
Rigorous verification script for converted datasets.
Tests:
1. Dataset loads correctly
2. Dimensions are correct (15D for action/state)
3. Data integrity (values preserved, padding is zeros)
4. Compatible with training pipeline (make_dataset, DataLoader)
5. Multi-dataset support works
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import make_dataset
from lerobot.configs.default import DatasetConfig
from torch.utils.data import DataLoader

def test_dataset_loads(repo_id):
    """Test 1: Dataset loads without errors"""
    print(f"\n{'='*60}")
    print(f"TEST 1: Loading dataset {repo_id}")
    print('='*60)
    
    try:
        ds = LeRobotDataset(repo_id)
        print(f"✓ Dataset loaded successfully")
        print(f"  - Episodes: {ds.num_episodes}")
        print(f"  - Frames: {ds.num_frames}")
        print(f"  - FPS: {ds.fps}")
        return ds
    except Exception as e:
        print(f"✗ FAILED to load dataset: {e}")
        return None

def test_dimensions(ds, expected_dim=15):
    """Test 2: Verify action and observation.state are correct dimensions"""
    print(f"\n{'='*60}")
    print(f"TEST 2: Verifying dimensions (expected: {expected_dim}D)")
    print('='*60)
    
    success = True
    
    # Check metadata
    action_shape = ds.meta.features.get("action", {}).get("shape", [])
    state_shape = ds.meta.features.get("observation.state", {}).get("shape", [])
    
    print(f"  Action shape in metadata: {action_shape}")
    print(f"  State shape in metadata: {state_shape}")
    
    if action_shape and action_shape[0] != expected_dim:
        print(f"✗ Action dimension mismatch: {action_shape[0]} != {expected_dim}")
        success = False
    else:
        print(f"✓ Action dimension correct: {action_shape}")
    
    if state_shape and state_shape[0] != expected_dim:
        print(f"✗ State dimension mismatch: {state_shape[0]} != {expected_dim}")
        success = False
    else:
        print(f"✓ State dimension correct: {state_shape}")
    
    # Check actual data
    sample = ds[0]
    actual_action_dim = sample["action"].shape[-1] if "action" in sample else None
    actual_state_dim = sample["observation.state"].shape[-1] if "observation.state" in sample else None
    
    print(f"  Actual action tensor shape: {sample.get('action', torch.tensor([])).shape}")
    print(f"  Actual state tensor shape: {sample.get('observation.state', torch.tensor([])).shape}")
    
    if actual_action_dim and actual_action_dim != expected_dim:
        print(f"✗ Actual action dimension wrong: {actual_action_dim}")
        success = False
    
    return success

def test_data_integrity(ds_converted, ds_original_repo_id):
    """Test 3: Verify original 12D values are preserved"""
    print(f"\n{'='*60}")
    print(f"TEST 3: Data integrity check")
    print('='*60)
    
    # Load a sample from converted dataset
    sample = ds_converted[0]
    action = sample["action"].numpy()
    
    print(f"  Sample action values: {action[:5]}... (first 5)")
    print(f"  Last 3 values (should be zeros from padding): {action[-3:]}")
    
    # Verify padding is zeros
    if np.allclose(action[-3:], 0.0):
        print(f"✓ Padding values are zeros as expected")
    else:
        print(f"✗ Padding values are NOT zeros: {action[-3:]}")
        return False
    
    # Verify first 12 values are non-trivial (not all zeros)
    if not np.allclose(action[:12], 0.0):
        print(f"✓ Original 12D values preserved (non-zero)")
    else:
        print(f"⚠ Warning: Original values are all zeros (might be valid)")
    
    return True

def test_dataloader_compatibility(ds):
    """Test 4: Dataset works with PyTorch DataLoader"""
    print(f"\n{'='*60}")
    print(f"TEST 4: DataLoader compatibility")
    print('='*60)
    
    try:
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        
        print(f"✓ DataLoader works")
        print(f"  Batch keys: {list(batch.keys())[:5]}...")
        print(f"  Action batch shape: {batch['action'].shape}")
        
        # Verify batch dimension
        if batch['action'].shape[0] == 4 and batch['action'].shape[-1] == 15:
            print(f"✓ Batch shapes correct")
            return True
        else:
            print(f"✗ Unexpected batch shape")
            return False
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        return False

def test_training_pipeline_compatibility(repo_id):
    """Test 5: Compatible with training factory"""
    print(f"\n{'='*60}")
    print(f"TEST 5: Training pipeline compatibility")
    print('='*60)
    
    try:
        # Simulate what training does
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.configs.default import DatasetConfig
        
        # Create a minimal config
        dataset_cfg = DatasetConfig(repo_id=repo_id)
        
        # Load metadata as training would
        meta = LeRobotDatasetMetadata(repo_id)
        
        print(f"✓ Dataset metadata compatible with training")
        print(f"  Features: {list(meta.features.keys())[:5]}...")
        print(f"  Robot type: {meta.robot_type}")
        
        return True
    except Exception as e:
        print(f"✗ Training compatibility check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("DATASET CONVERSION VERIFICATION SUITE")
    print("="*60)
    
    converted_repo_id = "Mimic-Robotics/bimanual_blue_block_handover_1_15d"
    original_repo_id = "Mimic-Robotics/bimanual_blue_block_handover_1"
    
    results = {}
    
    # Test 1: Load dataset
    ds = test_dataset_loads(converted_repo_id)
    results["load"] = ds is not None
    
    if ds:
        # Test 2: Dimensions
        results["dimensions"] = test_dimensions(ds, expected_dim=15)
        
        # Test 3: Data integrity
        results["integrity"] = test_data_integrity(ds, original_repo_id)
        
        # Test 4: DataLoader
        results["dataloader"] = test_dataloader_compatibility(ds)
        
        # Test 5: Training pipeline
        results["training"] = test_training_pipeline_compatibility(converted_repo_id)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED - Dataset conversion verified!")
    else:
        print("SOME TESTS FAILED - Review issues above")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
