#!/usr/bin/env python3
"""
Test the dataset conversion on a single dataset without pushing to Hub.
This allows us to validate the conversion locally before batch processing.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from convert_bimanual_complete import convert_dataset

def test_conversion():
    """Test conversion on bimanual_blue_block_handover_1"""
    
    print("\n" + "="*70)
    print("TESTING DATASET CONVERSION")
    print("="*70)
    print("\nThis will:")
    print("1. Download bimanual_blue_block_handover_1")
    print("2. Convert to new format (no Hub push)")
    print("3. Validate the output")
    print("\n" + "="*70 + "\n")
    
    try:
        # Run conversion locally without pushing
        output_dir = convert_dataset(
            input_repo_id="Mimic-Robotics/bimanual_blue_block_handover_1",
            output_repo_id="Mimic-Robotics/achal_mobile_bimanual_1_test",
            push_to_hub=False,
            keep_temp=True  # Keep for inspection
        )
        
        print("\n" + "="*70)
        print("VALIDATION")
        print("="*70)
        
        # Validate the conversion
        import json
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # Check metadata
        info_path = output_dir / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        
        print("\n[OK] Metadata checks:")
        print(f"  - Version: {info['codebase_version']}")
        print(f"  - Robot type: {info['robot_type']}")
        print(f"  - Action shape: {info['features']['action']['shape']}")
        print(f"  - Obs.state shape: {info['features']['observation.state']['shape']}")
        
        print("\n[OK] Camera checks:")
        cameras = [k for k in info['features'].keys() if 'observation.images' in k]
        for cam in cameras:
            cam_name = cam.replace('observation.images.', '')
            shape = info['features'][cam]['shape']
            print(f"  - {cam_name}: {shape[1]}x{shape[0]}")
        
        # Check video directories
        video_dir = output_dir / "videos"
        print("\n[OK] Video directory structure:")
        for cam_dir in sorted(video_dir.iterdir()):
            if cam_dir.is_dir():
                video_count = len(list(cam_dir.rglob("*.mp4")))
                print(f"  - {cam_dir.name}: {video_count} videos")
        
        # Try loading with LeRobotDataset
        print("\n[OK] Loading with LeRobotDataset...")
        try:
            dataset = LeRobotDataset("local_dataset", root=str(output_dir.parent), split="train")
            print(f"  - Loaded successfully: {len(dataset)} frames")
            
            # Check first frame
            sample = dataset[0]
            print(f"  - Action shape: {sample['action'].shape}")
            print(f"  - Obs.state shape: {sample['observation.state'].shape}")
            print(f"  - Cameras: {[k for k in sample.keys() if 'observation.images' in k]}")
            
        except Exception as e:
            print(f"  FAILED to load: {e}")
            return False
        
        print("\n" + "="*70)
        print("TEST PASSED - Conversion successful!")
        print(f"Output saved to: {output_dir}")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_conversion()
    sys.exit(0 if success else 1)
