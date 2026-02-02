#!/usr/bin/env python3
"""
Complete Steps 2-4 on an already v3.0-converted dataset.
Run this on /tmp/dataset_conversion_cvqdnpvv/Mimic-Robotics/bimanual_blue_block_handover_1
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from convert_bimanual_complete import (
    expand_actions_and_observations,
    rename_and_process_videos,
    update_metadata
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    dataset_path = Path("/tmp/dataset_conversion_cvqdnpvv/Mimic-Robotics/bimanual_blue_block_handover_1")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    logger.info(f"Completing conversion steps on: {dataset_path}")
    
    # Step 2: Expand action/observation dimensions
    data_dir = dataset_path / "data"
    expand_actions_and_observations(data_dir)
    
    # Step 3: Rename cameras and process videos
    video_dir = dataset_path / "videos"
    rename_and_process_videos(video_dir)
    
    # Step 4: Update metadata
    meta_dir = dataset_path / "meta"
    update_metadata(meta_dir)
    
    logger.info("="*70)
    logger.info("CONVERSION COMPLETE")
    logger.info(f"Dataset ready at: {dataset_path}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
