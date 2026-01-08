# Dataset Conversion: Bimanual to Mobile Bimanual

## Problem

**Old Datasets (v2.1):** 20 datasets with ~450 episodes total
- Action space: **12 dimensions** (6 joints × 2 arms, including grippers)
- Robot: Stationary bimanual (no mobile base)
- Format: LeRobot v2.1

**New Datasets (v3.0):** Recent mobile datasets
- Action space: **15 dimensions** (6 joints × 2 arms + 3 base velocities)
- Robot: Mobile bimanual (mecanum base + arms)
- Format: LeRobot v3.0

**Goal:** Combine old + new datasets for training ACT model

## Solution: Zero-Padding

### Action Space Mapping

```
OLD (12D):                          NEW (15D):
├─ left arm (6D)                    ├─ left arm (6D)
│  ├─ shoulder_pan                  │  ├─ shoulder_pan
│  ├─ shoulder_lift                 │  ├─ shoulder_lift  
│  ├─ elbow_flex                    │  ├─ elbow_flex
│  ├─ wrist_flex                    │  ├─ wrist_flex
│  ├─ wrist_roll                    │  ├─ wrist_roll
│  └─ gripper                       │  └─ gripper
├─ right arm (6D)                   ├─ right arm (6D)
│  ├─ shoulder_pan                  │  ├─ shoulder_pan
│  ├─ shoulder_lift                 │  ├─ shoulder_lift
│  ├─ elbow_flex                    │  ├─ elbow_flex
│  ├─ wrist_flex                    │  ├─ wrist_flex
│  ├─ wrist_roll                    │  ├─ wrist_roll
│  └─ gripper                       │  └─ gripper
                                    └─ base (3D)  ← ADDED
                                       ├─ vx = 0.0      (zero-padded)
                                       ├─ vy = 0.0      (zero-padded)
                                       └─ omega = 0.0   (zero-padded)
```

### Why Zero-Padding Works

1. **Physically Accurate:** Old robot had no base motion → base velocities were actually zero
2. **Model Learning:** ACT learns that base stays still for stationary tasks
3. **Backward Compatible:** Can still train on pure mobile tasks or mixed datasets

## Usage

### Convert Single Dataset

```bash
python3 src/mimic/scripts/convert_bimanual_to_mobile.py \
  --input-repo-id="Mimic-Robotics/bimanual_blue_block_handover_1" \
  --output-repo-id="Mimic-Robotics/mobile_bimanual_blue_block_handover_1"
```

### Convert All 20 Datasets (Batch)

```bash
cd /home/odin/mimic-lerobot
./src/mimic/scripts/batch_convert_datasets.sh
```

This will convert all 20 datasets:
- `bimanual_blue_block_handover_1` → `mobile_bimanual_blue_block_handover_1`
- `bimanual_blue_block_handover_2` → `mobile_bimanual_blue_block_handover_2`
- ... (18 more)

## Training with Mixed Datasets

Once converted, use multi-dataset training:

```yaml
# training_config.yaml
dataset:
  repo_id:
    # Old converted datasets (~450 episodes)
    - "Mimic-Robotics/mobile_bimanual_blue_block_handover_1"
    - "Mimic-Robotics/mobile_bimanual_blue_block_handover_2"
    # ... add all 20
    
    # New mobile datasets
    - "Mimic-Robotics/mimic_mobile_bimanual_handover_v1"
    - "Mimic-Robotics/mimic_mobile_bimanual_drift_v1"
    - "Mimic-Robotics/mimic_mobile_bimanual_drift_v2"
```

Or train command:

```bash
lerobot-train \
  policy=act \
  env=mimic_follower \
  dataset.repo_id="['Mimic-Robotics/mobile_bimanual_blue_block_handover_1','Mimic-Robotics/mimic_mobile_bimanual_handover_v1']"
```

## What The Script Does

1. **Downloads** old dataset from HuggingFace (v2.1 format)
2. **Updates metadata:**
   - Codebase version: v2.1 → v3.0
   - Action shape: [12] → [15]
   - Observation.state shape: [12] → [15]
   - Adds action names: base_vx, base_vy, base_omega
   - Adds state names: base_x, base_y, base_theta
3. **Converts data files:**
   - Reads each parquet file
   - Pads action arrays: (N, 12) → (N, 15) with zeros
   - Pads observation.state: (N, 12) → (N, 15) with zeros
   - Writes converted parquet files
4. **Copies videos** unchanged (same cameras in both setups)
5. **Pushes** to HuggingFace with new repo ID

## Verification

After conversion, verify dimensions:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

old_converted = LeRobotDataset("Mimic-Robotics/mobile_bimanual_blue_block_handover_1")
new_mobile = LeRobotDataset("Mimic-Robotics/mimic_mobile_bimanual_handover_v1")

print(f"Old converted action shape: {old_converted[0]['action'].shape}")  # [15]
print(f"New mobile action shape: {new_mobile[0]['action'].shape}")        # [15]

# Check base velocities are zero in converted dataset
print(f"Base velocities (should be ~0): {old_converted[0]['action'][-3:]}")  # [0, 0, 0]
```

## Dataset Statistics

### Before Conversion
- 20 datasets × ~25 episodes each = ~450 episodes
- Format: v2.1 (incompatible)
- Action space: 12D (incompatible)

### After Conversion
- 20 datasets × ~25 episodes each = ~450 episodes
- Format: v3.0 ✓
- Action space: 15D ✓
- Can be mixed with new mobile datasets ✓

## Alternative: Online Padding

Instead of converting datasets, you could pad during training:

```python
# In your dataset/dataloader code
def collate_fn(batch):
    for sample in batch:
        if sample['action'].shape[0] == 12:
            # Old dataset - pad with zeros
            sample['action'] = torch.cat([
                sample['action'], 
                torch.zeros(3)
            ])
    return default_collate(batch)
```

**However, pre-converting is recommended because:**
1. Cleaner - datasets are in consistent format
2. Faster - no runtime padding overhead
3. Reusable - converted datasets work for all future training
4. Compatible with multi-dataset training infrastructure

## Files Created

- `src/mimic/scripts/convert_bimanual_to_mobile.py` - Conversion script
- `src/mimic/scripts/batch_convert_datasets.sh` - Batch conversion
- `docs/DATASET_CONVERSION.md` - This guide
