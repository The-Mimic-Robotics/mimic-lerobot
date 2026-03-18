# Dataset Conversion Guide

## Overview

Two conversion approaches available:

### 1. Single Dataset Test (`convert_bimanual_complete.py`)
Convert one old dataset to verify the pipeline works.

```bash
python3 convert_bimanual_complete.py \
    --input-repo="Mimic-Robotics/bimanual_blue_block_handover_1" \
    --output-repo="Mimic-Robotics/test_mobile_bimanual_1" \
    --no-push \
    --keep-temp
```

### 2. Consolidate All Datasets (`consolidate_all_datasets.py`)  
**RECOMMENDED FOR PRODUCTION**: Merge all 21 old datasets into ONE large consolidated dataset.

```bash
python3 consolidate_all_datasets.py \
    --output-repo="Mimic-Robotics/mobile_bimanual_blue_block_handover_complete" \
    [--no-push] \
    [--keep-temp]
```

## What Gets Converted

### Input (21 old datasets):
- `bimanual_blue_block_handover_1-7, 14-26`
- `jetson_bimanual_recording_test`
- Format: v2.1, 12D actions/observations, old camera names

### Output (1 consolidated dataset):
- All episodes merged into single dataset
- Format: v3.0 with file consolidation
- 15D actions/observations (base_vx, base_vy, base_omega added)
- New camera names: `right_wrist`, `left_wrist`, `top`, `front`
- Videos: letterboxed top camera (640x480→1280x720), blank front camera
- Metadata: updated robot_type to `mimic_follower`

## Conversion Steps

The pipeline performs these transformations:

1. **Download & Merge** (consolidate only)
   - Downloads all 21 datasets
   - Merges episodes into single combined dataset
   - Reindexes episode numbers sequentially

2. **v2.1 → v3.0 Conversion**
   - Runs official LeRobot converter
   - Consolidates data/video files (1MB data threshold, 10MB video threshold)
   - Updates codebase_version metadata

3. **Dimension Expansion (12D → 15D)**
   - Zero-pads action and observation.state arrays
   - Adds base motion dimensions: [base_vx, base_vy, base_omega]

4. **Camera Renaming & Video Processing**
   - `wrist_right` → `right_wrist` (copy as-is)
   - `wrist_left` → `left_wrist` (copy as-is)  
   - `realsense_top` → `top` (letterbox 640x480 → 1280x720 with black bars)
   - Creates `front` camera with blank videos

5. **Metadata Updates**
   - robot_type: `mimic_follower`
   - Updated feature shapes and names for 15D arrays
   - Updated camera features with new names

## File Structure

### Before (v2.1 - Episode-based):
```
data/chunk-000/
  episode_000000.parquet
  episode_000001.parquet
  ...
videos/chunk-000/
  observation.images.wrist_right/
    episode_000000.mp4
  observation.images.wrist_left/
    episode_000000.mp4
  observation.images.realsense_top/
    episode_000000.mp4
```

### After (v3.0 - File-based):
```
data/chunk-000/
  file-000.parquet  # Consolidated (multiple episodes)
  file-001.parquet
  ...
videos/
  observation.images.right_wrist/chunk-000/
    file-000.mp4  # Consolidated
  observation.images.left_wrist/chunk-000/
    file-000.mp4
  observation.images.top/chunk-000/
    file-000.mp4  # Letterboxed
  observation.images.front/chunk-000/
    file-000.mp4  # Blank video
```

## Benefits of Consolidation

**Why merge all datasets into one:**

1. **Simpler Management**: One dataset vs 21 separate datasets
2. **Better Training**: Models train on full combined distribution
3. **Automatic Consolidation**: Large size triggers proper file-based v3.0 format
4. **Single Source of Truth**: No version confusion across multiple repos
5. **Easier Updates**: Update once instead of 21 times

**Expected Stats (consolidated):**
- ~525 total episodes (21 datasets × ~25 episodes each)
- ~320,000 frames total
- File-based consolidation will naturally occur due to large size

## Troubleshooting

### Issue: Video processing finds 0 cameras
**Cause**: Wrong path expectations  
**Fix**: Updated script now handles v3.0 converter's actual output structure

### Issue: Episode-based files instead of consolidated
**Cause**: Dataset too small for 100MB default threshold  
**Fix**: Use 1MB threshold for small datasets, or consolidate multiple datasets

### Issue: Token permission errors when deleting
**Cause**: HuggingFace token lacks write permissions  
**Fix**: Generate new token with write access at https://huggingface.co/settings/tokens

## Next Steps

1. **Test with one dataset** to verify conversion works
2. **Fix HuggingFace token** if needed for deletion
3. **Run consolidation script** to create the final merged dataset
4. **Verify output** by loading in LeRobot and checking shapes/cameras
5. **Delete old broken datasets** (mobile_bimanual_blue_block_handover_1-7, 14-26)
