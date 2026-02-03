#!/bin/bash
# Simple evaluation script using lerobot-record command
# Reads configuration from robot_config.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../src/mimic/config/robot_config.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Parse YAML config
TEMP_CONFIG=$(mktemp)
python3 << PYEOF > "$TEMP_CONFIG"
import yaml
import json

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

robot = config['robot']

# Build cameras JSON matching training format (right_wrist, left_wrist, front, top)
cameras = {}

if 'wrist_right' in robot['cameras']:
    cam = robot['cameras']['wrist_right']
    cameras['right_wrist'] = {
        'type': cam['type'],
        'index_or_path': cam['index_or_path'],
        'width': cam['width'],
        'height': cam['height'],
        'fps': cam['fps'],
    }
    if 'fourcc' in cam:
        cameras['right_wrist']['fourcc'] = cam['fourcc']

if 'wrist_left' in robot['cameras']:
    cam = robot['cameras']['wrist_left']
    cameras['left_wrist'] = {
        'type': cam['type'],
        'index_or_path': cam['index_or_path'],
        'width': cam['width'],
        'height': cam['height'],
        'fps': cam['fps'],
    }
    if 'fourcc' in cam:
        cameras['left_wrist']['fourcc'] = cam['fourcc']

if 'front' in robot['cameras']:
    cam = robot['cameras']['front']
    cameras['front'] = {
        'type': cam['type'],
        'index_or_path': cam['index_or_path'],
        'width': cam['width'],
        'height': cam['height'],
        'fps': cam['fps'],
    }
    if 'fourcc' in cam:
        cameras['front']['fourcc'] = cam['fourcc']

if 'head' in robot['cameras']:
    cam = robot['cameras']['head']
    cameras['top'] = {
        'type': 'zed_camera',
        'index_or_path': cam['index_or_path'],
        'width': cam['width'],
        'height': cam['height'],
        'fps': cam['fps'],
    }

cameras_json = json.dumps(cameras)

print(f'CAMERAS="{cameras_json}"')
print(f'ROBOT_TYPE="{robot["type"]}"')
print(f'ROBOT_ID="{robot["id"]}"')
print(f'ROBOT_LEFT_ARM="{robot["left_arm_port"]}"')
print(f'ROBOT_RIGHT_ARM="{robot["right_arm_port"]}"')
print(f'ROBOT_BASE="{robot["base_port"]}"')
PYEOF

source "$TEMP_CONFIG"
rm "$TEMP_CONFIG"

# Evaluation parameters
NUM_EPISODES=10
TASK="Navigate to table. Pick up blue block with closest arm. Transfer mid-air to opposite arm. Place in target area."
EVAL_DATASET="Mimic-Robotics/eval_act_multi_dataset_new_hands"
MODEL_PATH="outputs/train/act_multi_dataset_new_hands/checkpoints/020000/pretrained_model"

# Clean up existing dataset cache to avoid conflicts
CACHE_DIR="$HOME/.cache/huggingface/lerobot/$EVAL_DATASET"
if [ -d "$CACHE_DIR" ]; then
    echo "Cleaning up existing dataset cache at $CACHE_DIR"
    rm -rf "$CACHE_DIR"
fi

echo "============================================"
echo "Evaluating ACT Policy with lerobot-record"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Robot: $ROBOT_TYPE"
echo "Episodes: $NUM_EPISODES"
echo "Task: $TASK"
echo "Eval Dataset: $EVAL_DATASET"
echo "============================================"
echo ""

# Run lerobot-record for evaluation
lerobot-record \
  --robot.type=$ROBOT_TYPE \
  --robot.left_arm_port=$ROBOT_LEFT_ARM \
  --robot.right_arm_port=$ROBOT_RIGHT_ARM \
  --robot.base_port=$ROBOT_BASE \
  --robot.id=$ROBOT_ID \
  --robot.cameras="$CAMERAS" \
  --display_data=true \
  --dataset.repo_id=$EVAL_DATASET \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.single_task="$TASK" \
  --policy.path=$MODEL_PATH
