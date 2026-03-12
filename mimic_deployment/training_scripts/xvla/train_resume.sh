#!/bin/bash
# LeRobot Training Resume Script


# ./train_resume.sh /home/odin/mimic-lerobot/outputs/train/xvla_odin_full_ttt__xvla_nofr_2cam_50a_20260310_1213/checkpoints/last/pretrained_model/train_config.json



#fore ground 
# ./train_resume.sh /home/odin/mimic-lerobot/outputs/train/xvla_odin_full_ttt__xvla_nofr_2cam_50a_20260310_1213/checkpoints/last/pretrained_model/train_config.json --no-daemon
# 1. Check for the config path argument
if [ -z "$1" ]; then
    echo "❌ Error: Missing config path."
    echo "Usage: ./train_resume.sh <path/to/train_config.json> [--no-daemon]"
    exit 1
fi

CONFIG_PATH="$1"
DAEMON_FLAG="$2"

# 2. Optimize memory allocation for the RTX 3090
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_MIXED_PRECISION="bf16"

# 3. Setup workspace paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Note: Adjust REPO_ROOT depth if you place this script in a different directory
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)" 

cd "$REPO_ROOT" || { echo "Failed to cd to $REPO_ROOT"; exit 1; }

# Automatically extract the job name from the path for clean logging
# E.g., grabs "xvla_odin_full_ttt__xvla_nofr_2cam_50a_20260310_1213"
JOB_NAME=$(echo "$CONFIG_PATH" | awk -F'/checkpoints/' '{print $1}' | awk -F'/' '{print $NF}')
LOG_FILE="$REPO_ROOT/outputs/logs/${JOB_NAME}_resumed.log"

mkdir -p "$REPO_ROOT/outputs/logs"

echo "=========================================="
echo "🔄 Resuming X-VLA Training"
echo "Config:   $CONFIG_PATH"
echo "Log File: $LOG_FILE"
echo "=========================================="

# 4. The Resume Command
CMD=(python src/lerobot/scripts/lerobot_train.py \
  --config_path="$CONFIG_PATH" \
  --resume=true)

# 5. Execution Logic
if [[ "$DAEMON_FLAG" == "--no-daemon" ]]; then
    echo "Starting in FOREGROUND..."
    "${CMD[@]}"
else
    echo "Starting in BACKGROUND..."
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    echo "✅ Training resumed!"
    echo "Tail logs with: tail -f $LOG_FILE"
fi


