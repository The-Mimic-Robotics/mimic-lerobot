# Calibration Storage Location Change

## Summary

Changed the default calibration storage location from `~/.cache/huggingface/lerobot/calibration/` to `src/mimic/calibration/` to prevent loss of calibration data when cache directories are cleaned.

## Changes Made

### 1. Modified Constants File
**File**: [src/lerobot/utils/constants.py](../src/lerobot/utils/constants.py#L62-L66)

Changed the default calibration path calculation:

```python
# Before:
default_calibration_path = HF_LEROBOT_HOME / "calibration"

# After:
_mimic_calibration_path = Path(__file__).resolve().parent.parent.parent / "mimic" / "calibration"
default_calibration_path = _mimic_calibration_path
```

This change affects **all** robots and teleoperators in the system since:
- `Robot` class (in `src/lerobot/robots/robot.py`) uses `HF_LEROBOT_CALIBRATION`
- `Teleoperator` class (in `src/lerobot/teleoperators/teleoperator.py`) uses `HF_LEROBOT_CALIBRATION`
- All subclasses inherit this behavior automatically

### 2. Created New Directory Structure
```
src/mimic/calibration/
├── README.md              # Documentation
├── .gitkeep              # Ensures directory is tracked in git
├── robots/               # Follower robot calibrations
│   └── <robot_type>/
│       └── <robot_id>.json
└── teleoperators/        # Leader teleoperator calibrations
    └── <teleop_type>/
        └── <teleop_id>.json
```

### 3. Migrated Existing Calibrations
Copied all existing calibration files from the old location to the new one:
- `mimic_follower_left.json` and `mimic_follower_right.json` (robots)
- `mimic_leader_left.json` and `mimic_leader_right.json` (teleoperators)

## Why This Approach?

### Minimal Code Changes
Instead of modifying multiple robot/teleoperator classes, we changed **only one variable** in the constants file. This is the high-level utility that controls calibration storage for the entire system.

### How It Works
1. Both `Robot` and `Teleoperator` base classes import `HF_LEROBOT_CALIBRATION` from `constants.py`
2. In their `__init__` methods, they use this path to construct calibration file paths
3. All subclasses (so100_follower, mimic_follower, so100_leader, etc.) automatically inherit this behavior
4. By changing the constant, we affect the entire system with a single change

### Environment Variable Override
Users can still override the location by setting:
```bash
export HF_LEROBOT_CALIBRATION=/path/to/custom/calibration
```

## Benefits

1. **Persistence**: Calibrations survive cache cleanups
2. **Version Control**: Can be tracked in git if desired
3. **Portability**: Easy to backup with the project
4. **Accessibility**: Centralized in a logical location
5. **Minimal Changes**: Only one line changed in the codebase

## Testing

Verified that:
- ✓ Path resolves correctly to `src/mimic/calibration/`
- ✓ Existing calibration files were migrated successfully
- ✓ Directory structure is created properly
- ✓ Import system works without errors

## Future Calibrations

All new calibrations will automatically be saved to the new location. The system will:
1. Check for existing calibration at the new path
2. If not found, prompt for calibration
3. Save new calibration to `src/mimic/calibration/`

---

## Why They Used .cache Originally

The original developers chose `~/.cache/` for several reasons:

1. **XDG Base Directory Specification**: Follows Linux standards where `~/.cache` is the designated location for non-essential cached data

2. **HuggingFace Ecosystem**: The project uses HuggingFace's infrastructure, which already uses `~/.cache/huggingface/` for model weights, datasets, etc. It's a consistent pattern

3. **User-specific Data**: Cache directories are per-user, avoiding permission issues in multi-user systems

4. **Auto-cleanup Friendly**: System administrators often clean cache dirs, and the expectation is that cache data can be regenerated (which calibrations can be, through recalibration)

5. **Separation of Concerns**: Keeps runtime/generated data separate from source code

However, for calibration data specifically, this wasn't ideal because:
- Calibrations are time-consuming to regenerate
- They're device-specific and somewhat permanent
- Losing them causes operational disruption
- They're small files that don't benefit from cache cleanup

For datasets and models, `.cache` makes sense because they're large, can be re-downloaded, and benefit from cleanup. But calibrations are more like "configuration" than "cache" data.
