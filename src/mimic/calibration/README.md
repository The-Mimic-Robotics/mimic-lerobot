# Robot Calibration Files

This directory stores all robot and teleoperator calibration files for the mimic-lerobot project.

## Structure

```
calibration/
├── robots/          # Follower robot calibrations
│   └── <robot_type>/
│       └── <robot_id>.json
└── teleoperators/   # Leader teleoperator calibrations
    └── <teleop_type>/
        └── <teleop_id>.json
```

## Purpose

Calibration files contain motor-specific calibration data (offsets, drive modes, etc.) that are required for proper robot operation. These files are automatically created when you run the calibration process.

## Why This Location?

Previously, calibration files were stored in `~/.cache/huggingface/lerobot/calibration/`, which meant they could easily be lost when cache directories were cleaned. By storing them in the source tree under the `mimic` folder, we ensure:

1. **Persistence**: Calibrations survive cache cleanups
2. **Version Control**: Can be tracked in git if needed
3. **Portability**: Easy to backup and restore with the project
4. **Accessibility**: Centralized location within the project

## Override

You can still override the calibration directory by setting the `HF_LEROBOT_CALIBRATION` environment variable:

```bash
export HF_LEROBOT_CALIBRATION=/path/to/custom/calibration
```

## Usage

Calibration files are automatically loaded when connecting to robots/teleoperators. If a calibration file doesn't exist or needs updating, the system will prompt you to recalibrate.

To manually calibrate a device:

```bash
lerobot-calibrate --robot.type=<robot_type> --robot.id=<robot_id>
# or
lerobot-calibrate --teleop.type=<teleop_type> --teleop.id=<teleop_id>
```
