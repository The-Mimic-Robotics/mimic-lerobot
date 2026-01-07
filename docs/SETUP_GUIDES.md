# Mimic Robot Setup Guides

This document contains essential setup guides for working with the Mimic robot system.

---

## SSH GUI Display Setup

### Problem
When you SSH into a machine, GUI applications (rerun, RViz, etc.) fail with display errors or try to forward to your local machine instead of showing on the remote machine's physical monitor.

### Solution
Set the `DISPLAY` environment variable to `:1` to tell GUI apps to use the remote machine's physical display.

### One-Line Setup

Copy and run this command on the **remote machine** (via SSH):

```bash
echo '
# Auto-set DISPLAY for SSH sessions to use local monitor
if [ -n "$SSH_CONNECTION" ]; then
    export DISPLAY=:1
fi' >> ~/.bashrc && source ~/.bashrc && echo "✓ Setup complete! GUI apps will now appear on the remote monitor."
```

**Done!** All future SSH sessions will automatically use the remote display.

### Manual Setup

1. Add to your `~/.bashrc`:
   ```bash
   # Auto-set DISPLAY for SSH sessions to use local monitor
   if [ -n "$SSH_CONNECTION" ]; then
       export DISPLAY=:1
   fi
   ```

2. Apply changes:
   ```bash
   source ~/.bashrc
   ```

3. Test it:
   ```bash
   echo $DISPLAY  # Should output: :1
   ```

### Usage

After setup, just run your GUI commands normally via SSH:
```bash
./src/mimic/scripts/mimic-teleop  # Window appears on remote monitor
```

### Troubleshooting

**"Cannot open display"**
- Make sure X server is running on the remote machine
- Check active display: `ls /tmp/.X11-unix/` (usually shows `X1` = use `:1`)
- Try `:0` instead of `:1` if needed

**For current session only:**
```bash
export DISPLAY=:1
```

---

## GitHub Account Management

### Check Active Account
```bash
gh auth status
```
Look for "Active account: true" to see which account is active.

### Switch Accounts
```bash
gh auth switch --user <username>
```

Example:
```bash
gh auth switch --user ac-pate
gh auth switch --user BMathi9s
```

### Set Git Config for Your Account
Before pushing commits, set your identity:
```bash
git config user.name "your-username"
git config user.email "your-email@example.com"
```

### Workflow
1. **Switch to your account:**
   ```bash
   gh auth switch --user your-username
   ```

2. **Set your git config** (if not already set):
   ```bash
   git config user.name "your-username"
   git config user.email "your-email@example.com"
   ```

3. **Work and push:**
   ```bash
   git add .
   git commit -m "Your changes"
   git push
   ```

4. **After you're done:**
   - No need to log out
   - Next person should run `gh auth switch --user their-username`

### Important Notes
- Always verify active account with `gh auth status` before pushing
- The active account is used for all git push/pull operations
- No need to log out; just switch accounts when needed

---

## Persistent USB Device Names (udev rules)

The system uses udev rules to create persistent device names that don't change when you unplug/replug devices.

### Device Mappings

**Serial Devices (Arms and Base):**
- `/dev/arm_left_leader` → SO-100 Left Leader Arm
- `/dev/arm_right_leader` → SO-100 Right Leader Arm
- `/dev/arm_left_follower` → SO-100 Left Follower Arm
- `/dev/arm_right_follower` → SO-100 Right Follower Arm
- `/dev/mecanum_base` → Mecanum Base Controller (ESP32)

**Cameras:**
- `/dev/camera_left_wrist` → Left Wrist Camera
- `/dev/camera_right_wrist` → Right Wrist Camera
- `/dev/camera_zed` → ZED 2 Stereo Camera
- `/dev/camera_realsense` → Intel RealSense (if connected)

### Udev Rules Location
```bash
/etc/udev/rules.d/99-mimic-robot.rules
```

### How It Works
- **Arms:** Matched by CH340 serial numbers (unique per device)
- **Base:** Matched by vendor/product ID (1a86:7523)
- **Cameras:** Matched by vendor/product ID and video device index

### Reload Rules After Changes
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Verify Symlinks
```bash
ls -la /dev/arm_* /dev/mecanum_base /dev/camera_*
```

---

## Camera Configuration

### USB Bandwidth Optimization
The system uses **MJPEG compression** for wrist cameras to reduce USB bandwidth usage:
- Without MJPEG: ~350 Mbps for two 640×480@30fps cameras
- With MJPEG: <100 Mbps (allows ZED camera at 1280×720@30fps on same bus)

### ZED Camera Setup
- **Model:** ZED 2
- **Resolution:** 1280×720 (HD720)
- **Serial Number:** 23081456
- **SDK:** Uses pyzed.sl when available, falls back to OpenCV
- **Connection:** USB 3.0 required for HD720 resolution

### Camera Warmup
Set to `warmup_s: 0` in config to eliminate startup delay (was 3 seconds per camera = 9 second total delay).

---

## Quick Start Commands

### Teleoperation
```bash
./src/mimic/scripts/mimic-teleop
```

### Record Dataset
```bash
./src/mimic/scripts/mimic-record
```

### Replay Dataset
```bash
./src/mimic/scripts/mimic-replay
```

### Disable Cameras
```bash
./src/mimic/scripts/mimic-teleop --without-cameras
```

### Change Base Controller
Edit `src/mimic/config/robot_config.yaml`:
```yaml
teleop:
  base_control_mode: xbox  # or keyboard, joystick
```

---

## Troubleshooting

### Camera Timeout Issues
If cameras timeout during initialization:
1. Check USB bandwidth (lsusb -t)
2. Ensure MJPEG compression is enabled in config
3. Verify all cameras are detected: `ls /dev/video*`
4. Check for GStreamer interference (should be disabled in code)

### Mecanum Base Not Found
```bash
# Check if device exists
ls -la /dev/mecanum_base

# If missing, check physical connection
ls /dev/ttyUSB*

# Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### ZED Camera Not Detected
```bash
# Check SDK can see camera
python3 -c "import pyzed.sl as sl; cameras = sl.Camera.get_device_list(); print(f'Found {len(cameras)} cameras')"

# Check for processes using the camera
fuser /dev/video0 /dev/video1

# Kill stuck processes
kill <PID>
```

### Xbox Controller Not Connecting
```bash
# Check controller is detected
jstest /dev/input/js0

# List available gamepads
python3 -c "from inputs import devices; print(devices.gamepads)"
```
