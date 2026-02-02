# Quick Reference: Dataset Conversion

## Camera Naming

NO "head" camera exists. All datasets use:
- OLD: `realsense_top`
- NEW: `top`

## Dataset Status

### Need Conversion (21 datasets)
Old v2.1 format with old camera names:
```
bimanual_blue_block_handover_1, 2, 3, 4, 5, 6, 7
bimanual_blue_block_handover_14-26
jetson_bimanual_recording_test
```

### Properly Converted (7 datasets)
Correct v3.0 format with new camera names:
```
mimic_mobile_bimanual_drift_v2
mimic_displacement_to_handover_blue_block_v2, v6, v7, v8
mimic_displacement_to_handover_blue_block_with_new_hands_v2, v3
```

### Need RE-conversion (20 datasets)
Wrong v3.0 format - cameras NOT renamed:
```
mobile_bimanual_blue_block_handover_1, 2, 3, 4, 5, 6, 7
mobile_bimanual_blue_block_handover_14-26
```

## Scripts

### convert_bimanual_complete.py - WORKS
- Tested successfully
- Converts v2.1 -> v3.0
- Renames cameras correctly
- Adds front camera
- Letterboxes top to 1280x720

### batch_convert_datasets.sh - FIXED
- Updated to use convert_bimanual_complete.py
- Will convert all 21 old datasets
- Outputs to mobile_* repos

## Next Action
Re-convert the 20 broken mobile_bimanual datasets using convert_bimanual_complete.py
