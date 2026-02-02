# Dataset Format Comparison

Scan Date: 2026-02-02
Total Datasets: 48

## Summary

- **Old Format (v2.1, 12D)**: 21 datasets
- **New Format (v3.0, 15D)**: 27 datasets
- **No HEAD camera found**: All datasets use either "realsense_top" or "top"

## Key Findings

### 1. Old Format Datasets (v2.1, 12D action/obs)

All 21 old bimanual datasets have:
- Version: v2.1
- Robot Type: bi_so101_follower
- Action Dim: 12D (no base control)
- Obs State Dim: 12D (no base pose)
- Cameras: `wrist_right`, `wrist_left`, `realsense_top` (640x480)

List:
- bimanual_blue_block_handover_1 through 7
- bimanual_blue_block_handover_14 through 26
- jetson_bimanual_recording_test

### 2. New Format Datasets - Two Categories

#### A. Properly Converted (v3.0, modern names)

7 datasets with correct camera naming:
- mimic_mobile_bimanual_drift_v2
- mimic_displacement_to_handover_blue_block_v2
- mimic_displacement_to_handover_blue_block_v6
- mimic_displacement_to_handover_blue_block_v7
- mimic_displacement_to_handover_blue_block_v8
- mimic_displacement_to_handover_blue_block_with_new_hands_v2
- mimic_displacement_to_handover_blue_block_with_new_hands_v3

Configuration:
- Version: v3.0
- Robot Type: mimic_follower
- Action Dim: 15D (includes base_vx, base_vy, base_omega)
- Obs State Dim: 15D (includes base_x, base_y, base_theta)
- Cameras: `right_wrist`, `left_wrist`, `front`, `top` (top is 1280x720)

#### B. Incorrectly Converted (v3.0, old camera names)

20 datasets with WRONG camera naming:
- mobile_bimanual_blue_block_handover_1 through 7
- mobile_bimanual_blue_block_handover_14 through 26

Configuration:
- Version: v3.0
- Robot Type: mimic_follower (correct)
- Action Dim: 15D (correct)
- Obs State Dim: 15D (correct)
- Cameras: `wrist_right`, `wrist_left`, `realsense_top` (WRONG - old names)
- Missing: `front` camera, `top` renamed camera

**PROBLEM**: These datasets claim to be v3.0 but still use old v2.1 camera names. This is the "broken" state mentioned in CONVERSION_ANALYSIS_FEB_1.md.

## Camera Naming Summary

| Camera Type | Old Name | New Name | Found In |
|------------|----------|----------|----------|
| Right wrist | wrist_right | right_wrist | 7 properly converted datasets |
| Left wrist | wrist_left | left_wrist | 7 properly converted datasets |
| Top camera | realsense_top | top | 7 properly converted datasets |
| Front camera | (none) | front | 7 properly converted datasets (dummy/blank) |

**HEAD camera**: NOT FOUND in any dataset. All use either "realsense_top" (old) or "top" (new).

## Differences Between Old and New Formats

| Feature | Old (v2.1) | New (v3.0) |
|---------|-----------|-----------|
| Version | v2.1 | v3.0 |
| Robot Type | bi_so101_follower | mimic_follower |
| Action Dims | 12D | 15D (+base velocities) |
| Obs State Dims | 12D | 15D (+base pose) |
| Camera Count | 3 | 4 |
| Right Wrist | wrist_right | right_wrist |
| Left Wrist | wrist_left | left_wrist |
| Top Camera | realsense_top (640x480) | top (1280x720) |
| Front Camera | (none) | front (640x480, dummy) |
| Base Control | No | Yes (vx, vy, omega as zeros) |
| Base Pose | No | Yes (x, y, theta as zeros) |

## Conversion Script Status

### convert_bimanual_complete.py

**STATUS**: WORKS CORRECTLY

The script successfully:
- Upgrades v2.1 to v3.0
- Expands 12D to 15D with zero-padding
- Renames cameras: wrist_right -> right_wrist, wrist_left -> left_wrist, realsense_top -> top
- Creates dummy front camera (blank 640x480 videos)
- Letterboxes top camera from 640x480 to 1280x720
- Updates robot_type to mimic_follower
- Updates all metadata

Tested on: bimanual_blue_block_handover_1 (SUCCESS)

### Mobile Bimanual Datasets Issue

The 20 "mobile_bimanual_blue_block_handover_*" datasets appear to have been:
- Partially converted (dimensions expanded to 15D)
- BUT camera renaming was NOT done
- AND front camera was NOT added
- Results in broken/incompatible format

**Recommendation**: Re-convert all 20 mobile_bimanual datasets using convert_bimanual_complete.py

## Next Steps

1. Re-convert mobile_bimanual_blue_block_handover_* datasets (20 datasets)
2. Use convert_bimanual_complete.py script
3. Push properly converted datasets to replace broken ones
4. Update training configs to use correctly named cameras
