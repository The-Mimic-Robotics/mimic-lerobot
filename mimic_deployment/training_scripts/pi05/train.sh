#!/bin/bash
# Pi0.5 Training Script - Smart Configuration with Auto Normalization
# Updated to match Pi0 logic with Timestamped Job Names

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# CONFIGURATION 
# ============================================================================

POLICY_TYPE="pi05"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_RESOLVER="$REPO_ROOT/mimic_deployment/training_scripts/dataset_groups.py"

# Get computer name from environment or hostname
COMPUTER="${COMPUTER:-$(hostname)}"

# ============================================================================
# PARAMETERS (Override via command line arguments)
# ============================================================================

DATASET_GROUP="${DATASET_GROUP:-}"
SINGLE_DATASET="${SINGLE_DATASET:-}"

# Training parameters with computer-specific defaults
if [[ "$COMPUTER" == "odin" ]] || [[ "$COMPUTER" == "ODIN-IEEE" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
elif [[ "$COMPUTER" == "jupiter" ]] || [[ "$COMPUTER" == "mathias" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-1}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
else
    BATCH_SIZE="${BATCH_SIZE:-1}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
fi

STEPS="${STEPS:-150000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
ACTION_STEPS="${ACTION_STEPS:-50}" 
CHUNK_SIZE="${CHUNK_SIZE:-50}"

# ============================================================================
# RESOLVE DATASET
# ============================================================================

if [ ! -f "$DATASET_RESOLVER" ]; then
    echo "Error: Dataset resolver not found at $DATASET_RESOLVER"
    exit 1
fi

if [ -z "$DATASET_GROUP" ] && [ -z "$SINGLE_DATASET" ]; then
    echo "Error: Either DATASET_GROUP or SINGLE_DATASET must be provided"
    exit 1
fi

if [ -n "$DATASET_GROUP" ]; then
    DATASET_REPO_IDS=$(python3 "$DATASET_RESOLVER" "$DATASET_GROUP" --format bash)
    DATASET_NAME_FOR_JOB="$DATASET_GROUP"
else
    DATASET_REPO_IDS="$SINGLE_DATASET"
    DATASET_NAME_FOR_JOB=$(echo "$SINGLE_DATASET" | sed 's|.*/||')
fi

# ============================================================================
# AUTO-GENERATE TRAINING METADATA (WITH TIMESTAMP)
# ============================================================================

DATASET_NAME_CLEAN=$(echo "$DATASET_NAME_FOR_JOB" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

# Added Timestamp to Job Name to ensure unique logs/outputs
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}_${TIMESTAMP}"
fi

OUTPUT_DIR="$REPO_ROOT/outputs/train/${JOB_NAME}"
LOG_FILE="$REPO_ROOT/outputs/logs/${JOB_NAME}.log"
REPO_ID="Mimic-Robotics/${JOB_NAME}"

# ============================================================================
# TRAINING COMMAND
# ============================================================================

echo "=========================================="
echo "Pi0.5 Training Configuration"
echo "=========================================="
echo "Job Name:      $JOB_NAME"
echo "Computer:      $COMPUTER"
echo "Batch Size:    $BATCH_SIZE"
echo "Steps:         $STEPS"
echo "Log File:      $LOG_FILE"
echo "=========================================="

cd "$REPO_ROOT"

# pi0.5 uses different model architecture flags than pi0
CMD=(python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="$DATASET_REPO_IDS" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id="$REPO_ID" \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  # CRITICAL FOR 5070: Freezes the VLM, trains only the action head to save VRAM
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  # Input definition
  --policy.input_features='{
    "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.images.left_wrist": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.images.right_wrist": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.state": {"shape": [14], "type": "STATE"},
    "observation.instruction": {"type": "LANGUAGE", "shape": [1]}
  }' \
  --policy.device=cuda \
  --dataset.image_transforms.enable=false \
  --batch_size=12 \
  --num_workers="$NUM_WORKERS" \
  --steps=15000 \
  --save_freq=1000 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=true)

if [[ "$1" == "--no-daemon" ]]; then
    echo "Starting training in FOREGROUND..."
    "${CMD[@]}"
else
    echo "Starting training in BACKGROUND..."
    mkdir -p "$(dirname "$LOG_FILE")"
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$REPO_ROOT/outputs/logs/${JOB_NAME}.pid"
    echo "Training started with PID: $TRAIN_PID"
    echo "Monitor with: tail -f $LOG_FILE"
fi