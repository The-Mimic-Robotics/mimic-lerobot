#!/usr/bin/env python3
"""
Scan all Mimic-Robotics datasets on HuggingFace Hub to identify:
- Camera names (head vs top vs others)
- Action/observation dimensions
- Robot types
- Version differences
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def scan_dataset_info(repo_id: str) -> dict:
    """Download and parse info.json from a dataset."""
    try:
        info_path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/info.json",
            repo_type="dataset",
            cache_dir=None
        )
        
        with open(info_path) as f:
            info = json.load(f)
        
        # Extract camera names
        cameras = []
        for key in info.get('features', {}).keys():
            if 'observation.images' in key:
                cam_name = key.replace('observation.images.', '')
                shape = info['features'][key].get('shape', [])
                cameras.append({
                    'name': cam_name,
                    'resolution': f"{shape[1]}x{shape[0]}" if len(shape) >= 2 else "unknown"
                })
        
        return {
            'repo_id': repo_id,
            'version': info.get('codebase_version', 'unknown'),
            'robot_type': info.get('robot_type', 'unknown'),
            'action_dim': info.get('features', {}).get('action', {}).get('shape', [None])[0],
            'obs_state_dim': info.get('features', {}).get('observation.state', {}).get('shape', [None])[0],
            'cameras': cameras,
            'total_episodes': info.get('total_episodes', 0),
            'total_frames': info.get('total_frames', 0),
            'status': 'success'
        }
        
    except Exception as e:
        logger.warning(f"Failed to scan {repo_id}: {e}")
        return {
            'repo_id': repo_id,
            'status': 'failed',
            'error': str(e)
        }


def main():
    logger.info("Fetching Mimic-Robotics datasets from HuggingFace Hub...")
    
    api = HfApi()
    datasets = list(api.list_datasets(author="Mimic-Robotics"))
    
    logger.info(f"Found {len(datasets)} datasets")
    
    results = []
    for ds in tqdm(datasets, desc="Scanning datasets"):
        repo_id = ds.id
        result = scan_dataset_info(repo_id)
        results.append(result)
    
    # Categorize datasets
    old_format = []  # v2.1 bimanual
    new_format = []  # v3.0 mobile bimanual
    other = []
    
    camera_with_head = []
    camera_with_top = []
    
    for r in results:
        if r['status'] != 'success':
            continue
            
        # Check cameras
        cam_names = [c['name'] for c in r['cameras']]
        if 'head' in cam_names:
            camera_with_head.append(r['repo_id'])
        if 'top' in cam_names:
            camera_with_top.append(r['repo_id'])
        
        # Categorize by version and dimensions
        if r['version'] == 'v2.1' and r['action_dim'] == 12:
            old_format.append(r)
        elif r['version'] == 'v3.0' and r['action_dim'] == 15:
            new_format.append(r)
        else:
            other.append(r)
    
    # Save results
    output_file = Path(__file__).parent / "dataset_scan_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'total_datasets': len(datasets),
            'successful_scans': len([r for r in results if r['status'] == 'success']),
            'old_format_count': len(old_format),
            'new_format_count': len(new_format),
            'other_count': len(other),
            'datasets_with_head_camera': camera_with_head,
            'datasets_with_top_camera': camera_with_top,
            'all_results': results,
            'old_format': old_format,
            'new_format': new_format,
            'other': other
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("DATASET SCAN SUMMARY")
    print("="*70)
    print(f"Total datasets: {len(datasets)}")
    print(f"Successfully scanned: {len([r for r in results if r['status'] == 'success'])}")
    print(f"\nOld format (v2.1, 12D): {len(old_format)}")
    print(f"New format (v3.0, 15D): {len(new_format)}")
    print(f"Other: {len(other)}")
    print(f"\nDatasets with 'head' camera: {len(camera_with_head)}")
    print(f"Datasets with 'top' camera: {len(camera_with_top)}")
    print("="*70)


if __name__ == "__main__":
    main()
