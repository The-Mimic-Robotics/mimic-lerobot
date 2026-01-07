# Mimic Robot - Workflow Scripts Guide

This guide covers the YAML configuration and workflow scripts for simplified robot operations.

---

## Configuration

### Location
`src/mimic/config/robot_config.yaml`

### Overview
The config file stores all robot settings in one place. Edit this file to match your hardware setup.

### Configuration Sections

#### Robot Settings
```yaml
robot:
  type: mimic_follower
  id: mimic_follower
  left_arm_port: /dev/arm_left_follower
  right_arm_port: /dev/arm_right_follower
  base_port: /dev/mecanum_base
```

#### Camera Settings
```yaml
cameras:
  right_wrist:
    type: opencv
    index_or_path: /dev/camera_right_wrist
    width: 640
    height: 480
    fps: 30
  
  left_wrist:
    type: opencv
    index_or_path: /dev/camera_left_wrist
    width: 640
    height: 480
    fps: 30
  
  zed:
    type: zed_camera
    index_or_path: /dev/camera_zed
    width: 1280
    height: 720
    fps: 30
  
  realsense:
    type: opencv
    index_or_path: /dev/camera_realsense
    width: 640
    height: 480
    fps: 30
```

#### Teleoperation Settings
```yaml
teleop:
  type: mimic_leader
  id: mimic_leader
  left_arm_port: /dev/arm_left_leader
  right_arm_port: /dev/arm_right_leader
  base_control_mode: keyboard  # Options: keyboard, xbox, joystick
```

#### Dataset Settings
```yaml
dataset:
  repo_id: Mimic-Robotics/mimic_zed_data_test
  fps: 30
  num_image_writer_processes: 1
  video: true
  
  recording:
    num_episodes: 5
    episode_time_s: 60
    reset_time_s: 10
    single_task: "Default task description"
```

---

## Workflow Scripts

### Location
`src/mimic/scripts/`

All scripts automatically read settings from `../config/robot_config.yaml`.

---

### Teleoperation

**Basic (no cameras):**
```bash
./src/mimic/scripts/mimic-teleop
```

**With cameras:**
```bash
./src/mimic/scripts/mimic-teleop --cameras
```

**Description:** Runs teleoperation using all robot/teleop ports and settings from config file.

**Options:**
- `--cameras` or `--with-cameras` - Enable camera feeds

---

### Recording

**Basic (uses config defaults):**
```bash
./src/mimic/scripts/mimic-record
```

**Custom recording:**
```bash
./src/mimic/scripts/mimic-record --episodes 10 --episode-time 45 --task "Pick and place"
```

**Description:** Records demonstration episodes to HuggingFace dataset.

**Options:**
- `--episodes N` - Number of episodes to record (default: from config)
- `--episode-time S` - Episode duration in seconds (default: from config)
- `--reset-time S` - Reset time between episodes in seconds (default: from config)
- `--task "description"` - Task description (default: from config)
- `--repo repo_id` - Override dataset repo ID (default: from config)

**Examples:**
```bash
# Record 5 episodes with default settings
./src/mimic/scripts/mimic-record

# Record 20 episodes, 30 seconds each
./src/mimic/scripts/mimic-record --episodes 20 --episode-time 30

# Record with custom task description
./src/mimic/scripts/mimic-record --episodes 10 --task "Sorting objects"

# Record to different dataset
./src/mimic/scripts/mimic-record --repo MyOrg/my_dataset --task "New task"
```

---

### Replay

**Replay episode:**
```bash
./src/mimic/scripts/mimic-replay --episode 0
```

**Replay from different repo:**
```bash
./src/mimic/scripts/mimic-replay --episode 5 --repo Mimic-Robotics/other_dataset
```

**Description:** Replays recorded episode on the robot.

**Options:**
- `--episode N` - Episode number to replay (required)
- `--repo repo_id` - Override dataset repo ID (default: from config)

**Examples:**
```bash
# Replay first episode
./src/mimic/scripts/mimic-replay --episode 0

# Replay episode 10
./src/mimic/scripts/mimic-replay --episode 10

# Replay from different dataset
./src/mimic/scripts/mimic-replay --episode 3 --repo MyOrg/other_dataset
```

---

## Persistent Device Names

**Important:** The config file uses persistent device names like `/dev/arm_left_leader`. 

To set these up, run the setup script:
```bash
python3 mimic_deployment/scripts/setup_udev_rules.py
```

See the setup script's documentation (at the top of the file) for detailed instructions.

---

## Troubleshooting

**Config file not found:**
- Verify `robot_config.yaml` exists at `src/mimic/config/robot_config.yaml`
- Scripts look for config at `../config/robot_config.yaml` relative to script location

**Devices not found:**
- Verify persistent device names: `ls -l /dev/arm_* /dev/mecanum_* /dev/camera_*`
- If missing, run udev setup script (see "Persistent Device Names" above)
- Check udev rules: `cat /etc/udev/rules.d/99-mimic-robot.rules`

**Permission denied:**
- Make scripts executable: `chmod +x src/mimic/scripts/mimic-*`

**Camera settings:**
- Use persistent names: `/dev/camera_zed` (recommended)
- Or use video indices: `0`, `2`, `4` (may change on reconnection)

---

## Modifying Configuration

**To change device ports:**
1. Edit `src/mimic/config/robot_config.yaml`
2. Update port paths under `robot:` and `teleop:` sections
3. Save and run scripts normally

**To change camera settings:**
1. Edit camera section in config
2. Adjust resolution, FPS, or device paths
3. Test with `mimic-teleop --cameras`

**To change dataset settings:**
1. Edit `dataset:` section in config
2. Update repo ID, FPS, or recording parameters
3. Changes apply to next `mimic-record` run

**All changes take effect immediately - no need to restart anything.**
