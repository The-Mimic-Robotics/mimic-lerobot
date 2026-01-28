# first test teleop, make sure all camera good 
```bash
./src/mimic/scripts/mimic-teleop --cameras
```


# Mimic Recording Script - Quick Reference

```bash
/tmp/record_cmd.sh
```
## Basic Usage

```bash
./src/mimic/scripts/mimic-record   --task-name handover   --repo-tag mobile_bimanual_handover   --version 1   --episodes 5   --reset-time 5   --episode-time 60   --org Mimic-Robotics


# Record default blue block handover task (version 1)
./src/mimic/scripts/mimic-record --task-name blue_block_handover --version 1

# Record version 2 of same task
./src/mimic/scripts/mimic-record --task-name blue_block_handover --version 2

# Custom number of episodes
./src/mimic/scripts/mimic-record --task-name blue_block_handover --version 1 --episodes 10

# Custom timing
./src/mimic/scripts/mimic-record --task-name displacement_to_grab_blue --version 1 \
  --episodes 20 --episode-time 60 --reset-time 5
```

## Dataset Naming Convention

Datasets are automatically named using the pattern:
```
{org}/mobile_bimanual_{task-name}_v{version}
```

Examples:
- `Mimic-Robotics/mobile_bimanual_blue_block_handover_v1`
- `Mimic-Robotics/mobile_bimanual_blue_block_handover_v2`
- `Mimic-Robotics/mobile_bimanual_custom_task_v1`

## All Available Arguments

### Required
- `--task-name NAME` - Short task identifier (e.g., "blue_block_handover")
- `--version N` - Dataset version number (1, 2, 3, ...)

### Optional
- `--episodes N` - Number of episodes to record (default: from config)
- `--episode-time S` - Seconds per episode (default: from config)
- `--reset-time S` - Seconds between episodes (default: from config)
- `--task-desc "TEXT"` - Custom VLA task description (default: auto-generated)
- `--repo-tag TAG` - Custom repo suffix (default: same as task-name)
- `--org NAME` - HuggingFace org (default: Mimic-Robotics)
- `--fps N` - Recording FPS (default: from config)
- `--writers N` - Parallel image writers (default: from config)
- `--no-video` - Disable video encoding
- `--no-display` - Disable rerun visualization
- `--help` - Show help

## Task Descriptions

The script auto-generates VLA-optimized task descriptions. For the default blue block handover:

**Auto-generated prompt:**
```
Navigate to table front by strafing left while turning clockwise. 
Pick up the blue block with the arm closest to it. Transfer the 
block mid-air to the opposite arm. Place the block in the 
designated target area using the receiving arm.
```

**Custom prompt:**
```bash
./src/mimic/scripts/mimic-record --task-name my_task --version 1 \
  --task-desc "Pick red cube from left shelf, place on right table"
```

## Current Task: Blue Block Handover

**Setup:**
- Robot starts at side of rectangular table
- Blue block is placed on table (left or right side)
- Target area is marked on table

**Execution:**
1. **Navigate**: Robot strafes left while turning clockwise to face table front
2. **Pick**: Closer arm picks up blue block based on block position
   - Block on right → right arm picks
   - Block on left → left arm picks
3. **Transfer**: Mid-air hand-off to opposite arm
4. **Place**: Receiving arm places block in target area

**Duration:** ~30-60 seconds per episode
**Success criteria:** Block placed securely in target zone

## Tips

1. **Versioning**: Increment version for:
   - Different block positions
   - Lighting changes
   - Hardware modifications
   - Improved demonstrations

2. **Episode Count**: Start with 5-10 episodes for testing, 50-100+ for training

3. **Episode Time**: Allow enough time for:
   - Navigation (~10s)
   - Pick (~5-10s)
   - Transfer (~5s)
   - Place (~5-10s)
   - Buffer (~10s)
   - Total: 30-60s recommended

4. **Reset Time**: Allow enough for:
   - Return block to start position
   - Reset robot to side position
   - 10-15s recommended

## Example Workflows

### Quick test recording (5 episodes)
```bash
./src/mimic/scripts/mimic-record --task-name blue_block_handover --version 1 \
  --episodes 5 --episode-time 30 --reset-time 10
```

### Production recording (50 episodes)
```bash
./src/mimic/scripts/mimic-record --task-name blue_block_handover --version 1 \
  --episodes 50 --episode-time 45 --reset-time 15
```

### Continued recording (version 2)
```bash
./src/mimic/scripts/mimic-record --task-name blue_block_handover --version 2 \
  --episodes 50
```

### Custom task
```bash
./src/mimic/scripts/mimic-record --task-name custom_assembly --version 1 \
  --episodes 20 \
  --task-desc "Navigate to workbench. Pick bolt with right arm. Thread onto shaft held by left arm."
```

## Troubleshooting

**Problem:** Script can't find config file
- **Solution:** Run from mimic-lerobot root directory

**Problem:** Camera not detected
- **Solution:** Check `/dev/camera_*` symlinks exist, re-run udev setup

**Problem:** Controller not working
- **Solution:** Verify Xbox controller connected, check `./src/mimic/scripts/mimic-teleop`

**Problem:** Upload to HuggingFace fails
- **Solution:** Run `huggingface-cli login` and authenticate

## See Also

- `task_prompts.yaml` - VLA-optimized task descriptions
- `robot_config.yaml` - Hardware configuration
- `xbox.yaml` - Controller settings
