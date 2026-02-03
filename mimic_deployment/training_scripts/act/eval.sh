#!/bin/bash
# ACT Evaluation Script - Smart Configuration
# Automatically configures evaluation based on robot config and environment variables

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

POLICY_TYPE="act"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$REPO_ROOT/src/mimic/config/robot_config.yaml"

# ============================================================================
# REQUIRED ENVIRONMENT VARIABLES (Set these before running)
# ============================================================================

: ${MODEL_PATH:?"Error: MODEL_PATH not set. Example: export MODEL_PATH=Mimic-Robotics/act_odin_all_datasets"}
: ${EVAL_DATASET:?"Error: EVAL_DATASET not set. Example: export EVAL_DATASET=Mimic-Robotics/eval_act_test"}
: ${NUM_EPISODES:=10}
: ${TASK:="Bimanual blue block handover from left to right hand"}

# ============================================================================
# PARSE ROBOT CONFIG
# ============================================================================

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Robot config not found at $CONFIG_FILE"
    exit 1
fi

echo "Parsing robot configuration from $CONFIG_FILE"

# Parse YAML and extract robot configuration
TEMP_CONFIG=$(mktemp)
python3 << PYEOF > "$TEMP_CONFIG"
import yaml
import json

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

robot = config['robot']
cameras_json = json.dumps(robot['cameras'])

print(f'ROBOT_TYPE="{robot["type"]}"')
print(f'ROBOT_ID="{robot["id"]}"')
print(f'ROBOT_LEFT_ARM="{robot["left_arm_port"]}"')
print(f'ROBOT_RIGHT_ARM="{robot["right_arm_port"]}"')
print(f'ROBOT_BASE="{robot["base_port"]}"')
print(f'CAMERAS="{cameras_json}"')
PYEOF

# Source the config
source "$TEMP_CONFIG"
rm "$TEMP_CONFIG"

# ============================================================================
# EVALUATION COMMAND
# ============================================================================

echo "=========================================="
echo "ACT Evaluation Configuration"
echo "=========================================="
echo "Model Path:    $MODEL_PATH"
echo "Robot Type:    $ROBOT_TYPE"
echo "Robot ID:      $ROBOT_ID"
echo "Dataset:       $EVAL_DATASET"
echo "Episodes:      $NUM_EPISODES"
echo "Task:          $TASK"
echo "=========================================="
echo ""

cd "$REPO_ROOT"

lerobot-record \
  --robot.type="$ROBOT_TYPE" \
  --robot.left_arm_port="$ROBOT_LEFT_ARM" \
  --robot.right_arm_port="$ROBOT_RIGHT_ARM" \
  --robot.base_port="$ROBOT_BASE" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="$CAMERAS" \
  --display_data=true \
  --dataset.repo_id="$EVAL_DATASET" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.single_task="$TASK" \
  --policy.path="$MODEL_PATH"

echo ""
echo "Evaluation complete!"
echo "Results saved to: $EVAL_DATASET"
