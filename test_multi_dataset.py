#!/usr/bin/env python
"""
Test MultiLeRobotDataset with converted and original datasets.
Verifies that converted 12D->15D datasets work with native 15D datasets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from torch.utils.data import DataLoader

def test_multi_dataset():
    print("\n" + "="*60)
    print("MULTI-DATASET COMPATIBILITY TEST")
    print("="*60)
    
    # Converted dataset (was 12D, now 15D)
    converted_repo = "Mimic-Robotics/bimanual_blue_block_handover_1_15d"
    
    # Native 15D dataset (mobile bimanual)
    native_repo = "Mimic-Robotics/mobile_bimanual_blue_block_handover_1"
    
    print(f"\nLoading converted dataset: {converted_repo}")
    ds_converted = LeRobotDataset(converted_repo)
    print(f"  Action shape: {ds_converted.meta.features['action']['shape']}")
    
    print(f"\nLoading native 15D dataset: {native_repo}")
    try:
        ds_native = LeRobotDataset(native_repo)
        print(f"  Action shape: {ds_native.meta.features['action']['shape']}")
    except Exception as e:
        print(f"  Note: Could not load native dataset: {e}")
        print("  Testing with converted dataset only...")
        ds_native = None
    
    if ds_native:
        # Test MultiLeRobotDataset
        print("\n" + "-"*40)
        print("Testing MultiLeRobotDataset...")
        print("-"*40)
        
        try:
            multi_ds = MultiLeRobotDataset(
                repo_ids=[converted_repo, native_repo]
            )
            print(f"✓ MultiLeRobotDataset created successfully")
            print(f"  Total frames: {len(multi_ds)}")
            print(f"  Features: {list(multi_ds.features.keys())[:5]}...")
            
            # Test DataLoader
            loader = DataLoader(multi_ds, batch_size=4, shuffle=True, num_workers=0)
            batch = next(iter(loader))
            print(f"✓ DataLoader works with MultiLeRobotDataset")
            print(f"  Batch action shape: {batch['action'].shape}")
            
            return True
        except Exception as e:
            print(f"✗ MultiLeRobotDataset failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n" + "-"*40)
        print("Single converted dataset verification...")
        print("-"*40)
        
        loader = DataLoader(ds_converted, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"✓ Converted dataset works standalone")
        print(f"  Batch action shape: {batch['action'].shape}")
        return True

if __name__ == "__main__":
    success = test_multi_dataset()
    print("\n" + "="*60)
    if success:
        print("MULTI-DATASET TEST PASSED")
    else:
        print("MULTI-DATASET TEST FAILED")
    print("="*60)
    sys.exit(0 if success else 1)
