# Multi-Dataset Training Guide

## Overview

The mimic-lerobot repository now supports training on multiple datasets simultaneously. This is useful for:
- Training on data collected across different sessions
- Combining datasets with different environmental conditions
- Leveraging diverse demonstrations for more robust policies

## How It Works

When multiple datasets are provided, the system:
1. Concatenates all datasets into a single virtual dataset
2. Merges episode metadata with global indexing
3. Aggregates normalization statistics across all datasets
4. Maintains per-sample `dataset_index` to track data source

## Usage

### Configuration

In your training config, provide a list of dataset repo_ids instead of a single string:

```yaml
# Single dataset (existing behavior)
dataset:
  repo_id: "Mimic-Robotics/mimic_mobile_bimanual_drift_v1"

# Multi-dataset training (new feature)
dataset:
  repo_id:
    - "Mimic-Robotics/mimic_mobile_bimanual_drift_v1"
    - "Mimic-Robotics/mimic_mobile_bimanual_drift_v2"
    - "Mimic-Robotics/mimic_mobile_bimanual_handover_v1"
```

### Python API

```python
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig

# Load config with multiple datasets
cfg = TrainPipelineConfig.from_yaml("path/to/config.yaml")
cfg.dataset.repo_id = [
    "Mimic-Robotics/mimic_mobile_bimanual_drift_v1",
    "Mimic-Robotics/mimic_mobile_bimanual_drift_v2",
]

# Create multi-dataset (automatic detection)
dataset = make_dataset(cfg)

# Access merged metadata
print(f"Total episodes: {dataset.num_episodes}")
print(f"Total frames: {dataset.num_frames}")
print(f"Dataset mapping: {dataset.repo_id_to_index}")
```

### Training Script

```bash
# Train with multiple datasets
lerobot-train \
  policy=act \
  env=mimic_follower \
  dataset.repo_id="['Mimic-Robotics/mimic_mobile_bimanual_drift_v1','Mimic-Robotics/mimic_mobile_bimanual_drift_v2']"
```

## Requirements

All datasets must:
- **Have compatible features** - Only common features across all datasets are kept
- **Same FPS** - Frame rate should match (verified automatically)
- **Same structure** - delta_timestamps computed from first dataset applies to all
- **Same robot type** - For proper action/observation dimensionality

## Data Structure

Each batch sample includes:
- `episode_index`: Global episode number (0 to total_episodes-1)
- `dataset_index`: Which source dataset (0 to num_datasets-1)
- Standard observation/action tensors

Example:
```python
for batch in dataloader:
    # batch["episode_index"] - global episode across all datasets
    # batch["dataset_index"] - which dataset this sample came from
    # batch["observation.state"], batch["action"], etc.
```

## Normalization

Statistics are automatically aggregated across all datasets:
- Min/max values are computed globally
- Mean/std are weighted by dataset size
- Per-camera statistics are merged

To use ImageNet stats instead:
```yaml
dataset:
  use_imagenet_stats: true
```

## Episode Indexing

Episodes are renumbered globally:
```
Dataset 1: Episodes 0-9   → Global 0-9
Dataset 2: Episodes 0-14  → Global 10-24
Dataset 3: Episodes 0-4   → Global 25-29
```

## Debugging

Check which features were disabled:
```python
dataset = make_dataset(cfg)
print(f"Disabled features: {dataset.disabled_features}")
```

Common issues:
- **"No keys common to all datasets"** - Datasets have completely different features
- **Features disabled warning** - Some features exist in only some datasets (automatic)

## Implementation Details

### Files Modified
- `src/lerobot/datasets/factory.py` - Enables multi-dataset creation
- `src/lerobot/datasets/lerobot_dataset.py` - Enhanced MultiLeRobotDataset class

### Key Components
1. **Metadata Merging** - Episodes, stats, and info merged from all datasets
2. **Global Indexing** - Episode indices remapped to avoid collisions
3. **Feature Filtering** - Only common features across datasets are kept
4. **Statistics Aggregation** - Min/max/mean/std computed globally

### Backward Compatibility
Single-dataset training unchanged:
- Passing a string `repo_id` creates `LeRobotDataset` (existing behavior)
- Passing a list of `repo_id` creates `MultiLeRobotDataset` (new feature)

## Performance

Multi-dataset training has similar performance to single-dataset:
- No additional memory overhead per sample
- Minimal CPU overhead for index mapping
- Same disk I/O patterns as single dataset

## Examples

### Training on Multiple Versions
```yaml
dataset:
  repo_id:
    - "Mimic-Robotics/mimic_mobile_bimanual_handover_v1"
    - "Mimic-Robotics/mimic_mobile_bimanual_handover_v2"
    - "Mimic-Robotics/mimic_mobile_bimanual_handover_v3"
```

### Combining Different Tasks
```yaml
dataset:
  repo_id:
    - "Mimic-Robotics/mimic_mobile_bimanual_handover_v1"
    - "Mimic-Robotics/mimic_mobile_bimanual_sorting_v1"
    - "Mimic-Robotics/mimic_mobile_bimanual_placement_v1"
```

### Filtering Episodes Per Dataset
```yaml
dataset:
  repo_id:
    - "Mimic-Robotics/dataset1"
    - "Mimic-Robotics/dataset2"
  episodes:
    "Mimic-Robotics/dataset1": [0, 1, 2, 3, 4]  # Use only first 5 episodes
    "Mimic-Robotics/dataset2": null  # Use all episodes
```

## Future Enhancements

Potential improvements (not yet implemented):
- [ ] Per-robot normalization for mixed robot types
- [ ] Weighted sampling across datasets
- [ ] Dataset-specific delta_timestamps
- [ ] Streaming multi-dataset support

## References

- Ported from `bimanual-lerobot` repository
- Compatible with LeRobot v3.0 codebase
- Based on PyTorch's `ConcatDataset` concept
