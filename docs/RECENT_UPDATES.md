# Mimic Robot System - Recent Updates (Jan 7, 2026)

## Summary
Successfully integrated ZED SDK, optimized camera bandwidth, implemented base control system, and created streamlined configuration workflow.

## Major Changes

### 1. Camera System Improvements

**OpenCV GStreamer Fix** (`f5bfebd4`)
- Disabled GStreamer backend that was causing timeout issues
- Critical fix borrowed from bimanual-lerobot repository
- Resolves camera connection failures and frame grab timeouts

**ZED SDK Integration** (`5fd208af`)
- Full hybrid implementation: uses pyzed.sl SDK when available, falls back to OpenCV
- Identifies ZED camera by serial number (23081456) for robust reconnection
- Supports HD720 resolution (1280×720@30fps) via SDK
- Proper thread management to prevent segfaults on disconnect
- Auto-detects SDK availability at import time

**USB Bandwidth Optimization** (`fb826ac2`)
- Added MJPEG compression for wrist cameras
- Reduces bandwidth from ~350 Mbps to <100 Mbps
- Enables simultaneous operation of 3 cameras without timeouts
- Eliminated 9-second startup delay (warmup_s: 3 → 0)

### 2. Base Control System

**Controller Architecture** (`6e0732fe`)
- Abstract `BaseController` interface
- Three implementations:
  - `KeyboardBaseController` - WASD+QE controls
  - `XboxBaseController` - Xbox Series S|X controller support
  - `JoystickBaseController` - Alternative gamepad support
- Configurable speeds, deadzone, and button mappings
- ROS2 joy_node compatible configuration format

**Teleoperation Integration** (`686100a1`)
- Integrated base control with bimanual SO-100 arm teleoperation
- Combined arm actions with base velocity commands
- Configurable mode selection (keyboard/xbox/joystick)
- Safety-ready architecture (safety button support ready to implement)

### 3. Configuration System

**Centralized Configuration** (`fb826ac2`, `8a737402`)
- Created `src/mimic/config/robot_config.yaml` - single source of truth
- Controller-specific configs: `xbox.yaml`, `joystick.yaml`, `keyboard.yaml`
- ROS2 joy_node parameter format for familiarity

**Wrapper Scripts** (`8a737402`)
- `mimic-teleop` - Start teleoperation
- `mimic-record` - Record datasets
- `mimic-replay` - Replay recorded data
- Auto-load YAML configs and build command-line arguments
- Simple usage: `./src/mimic/scripts/mimic-teleop` instead of long Python commands

### 4. Persistent USB Device Mapping

**Robust udev Rules**
- Matches devices purely by vendor/product ID (no USB port dependency)
- Arms identified by unique CH340 serial numbers
- Base matches by vendor 1a86, product 7523
- Cameras match by vendor/product and video index
- Symlinks persist across reconnections: `/dev/mecanum_base`, `/dev/arm_*`, `/dev/camera_*`

### 5. Documentation

**Consolidated Guides** (`275f4944`)
- Created `docs/SETUP_GUIDES.md` with all essential information
- Includes: SSH GUI setup, GitHub account management, USB device mapping
- Added camera configuration and troubleshooting sections
- Quick start commands and workflow guide
- Removed scattered/duplicate documentation

## System Status

### ✅ Working Features
- ZED camera at 1280×720@30fps via SDK
- All 3 cameras running simultaneously with MJPEG
- Mecanum base control (Xbox controller verified)
- Bimanual SO-100 arm teleoperation
- Persistent USB device names
- Fast startup (no warmup delay)
- Clean shutdown (no segfaults)

### 🏗️ Ready for Implementation
- Xbox safety button (RB) for base motion enable/disable
- Dataset recording with full camera suite
- Training pipeline with new camera setup

## Performance Metrics
- **Teleoperation loop:** ~30 Hz
- **Camera startup:** <2 seconds (was ~11 seconds)
- **USB bandwidth:** <200 Mbps total (was hitting 480 Mbps limit)
- **ZED resolution:** 1280×720 (was 672×376 with OpenCV fallback)

## Configuration Examples

### Robot Config
```yaml
cameras:
  right_wrist:
    type: opencv
    fourcc: MJPG  # Bandwidth optimization
    width: 640
    height: 480
    fps: 30
    warmup_s: 0
  top:
    type: zed_camera
    index_or_path: 23081456  # Serial number
    width: 1280
    height: 720
    fps: 30

teleop:
  base_control_mode: xbox
  max_linear_speed: 0.1  # m/s
  max_angular_speed: 0.2  # rad/s
```

### Usage
```bash
# Teleoperation
./src/mimic/scripts/mimic-teleop

# Without cameras (for debugging)
./src/mimic/scripts/mimic-teleop --without-cameras

# Recording
./src/mimic/scripts/mimic-record
```

## Git Commits
```
275f4944 docs: consolidate setup guides and remove legacy documentation
8a737402 feat(scripts): add configuration and wrapper scripts
686100a1 feat(teleop): integrate base control with bimanual leader
6e0732fe feat(teleop): add base controller system for mecanum platform
fb826ac2 feat(config): optimize camera and performance settings
5fd208af feat(cameras): add ZED SDK integration with OpenCV fallback
f5bfebd4 fix(cameras): disable GStreamer backend for OpenCV cameras
```

## Next Steps
1. Implement Xbox safety button for base motion control
2. Test dataset recording with full camera suite
3. Verify MJPEG quality for training data
4. Document training pipeline with new setup
