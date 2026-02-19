#!/bin/bash
# Pi0-FAST Training Script - Adapted for Tic-Tac-Toe
# Automatically configures training based on $COMPUTER environment variable

set -e

# Pi0-FAST benefits heavily from bf16 and gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_MIXED_PRECISION="bf16"

# ============================================================================
# CONFIGURATION
# ============================================================================

POLICY_TYPE="pi0_fast"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_RESOLVER="$REPO_ROOT/mimic_deployment/training_scripts/dataset_groups.py"

# Get computer name from environment or hostname
COMPUTER="${COMPUTER:-$(hostname)}"

# ============================================================================
# PARAMETERS (Override via command line arguments)
# ============================================================================

# Dataset group (required)
DATASET_GROUP="${DATASET_GROUP:-}"
SINGLE_DATASET="${SINGLE_DATASET:-}"

# Training parameters adapted for Pi0-FAST on 3090 Ti
if [[ "$COMPUTER" == "odin" ]] || [[ "$COMPUTER" == "ODIN-IEEE" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-4}" 
    NUM_WORKERS="${NUM_WORKERS:-8}"
elif [[ "$COMPUTER" == "mathias" ]]; then
    # 3090 Ti (24GB) constraint: Pi0-FAST is heavier than SmolVLA due to SigLIP+Gemma
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
else
    BATCH_SIZE="${BATCH_SIZE:-1}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
fi

# Libero benchmark used 10k-40k steps for finetuning. 30k is a safe middle ground.
STEPS="${STEPS:-100000}" 
SAVE_FREQ="${SAVE_FREQ:-10000}"

# Pi0-FAST Defaults (from Documentation)
ACTION_STEPS="${ACTION_STEPS:-10}" # [cite: 215]
CHUNK_SIZE="${CHUNK_SIZE:-10}"     # [cite: 214]

# ============================================================================
# RESOLVE DATASET GROUP TO DATASET LIST OR USE SINGLE DATASET
# ============================================================================

if [ ! -f "$DATASET_RESOLVER" ]; then
    echo "Error: Dataset resolver not found at $DATASET_RESOLVER"
    exit 1
fi

if [ -z "$DATASET_GROUP" ] && [ -z "$SINGLE_DATASET" ]; then
    echo "Error: Either DATASET_GROUP or SINGLE_DATASET must be provided"
    exit 1
fi

if [ -n "$DATASET_GROUP" ] && [ -n "$SINGLE_DATASET" ]; then
    echo "Error: Cannot specify both DATASET_GROUP and SINGLE_DATASET"
    exit 1
fi

if [ -n "$DATASET_GROUP" ]; then
    echo "Resolving dataset group: $DATASET_GROUP"
    DATASET_REPO_IDS=$(python3 "$DATASET_RESOLVER" "$DATASET_GROUP" --format bash)

    if [ $? -ne 0 ]; then
        echo "Error: Failed to resolve dataset group '$DATASET_GROUP'"
        python3 "$DATASET_RESOLVER" --list-groups
        exit 1
    fi

    DATASET_NAME_FOR_JOB="$DATASET_GROUP"
else
    echo "Using single dataset: $SINGLE_DATASET"
    DATASET_REPO_IDS="$SINGLE_DATASET"
    DATASET_NAME_FOR_JOB=$(echo "$SINGLE_DATASET" | sed 's|.*/||')
fi

echo "Datasets: $DATASET_REPO_IDS"

# ============================================================================
# AUTO-GENERATE TRAINING METADATA
# ============================================================================

DATASET_NAME_CLEAN=$(echo "$DATASET_NAME_FOR_JOB" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"
fi
OUTPUT_DIR="$REPO_ROOT/outputs/train/${JOB_NAME}"
LOG_FILE="$REPO_ROOT/outputs/logs/${JOB_NAME}.log"

REPO_ID="Mimic-Robotics/${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"

if [ -n "$DATASET_GROUP" ]; then
    WANDB_NOTES="Multi-dataset Pi0-FAST training on ${DATASET_GROUP} with Computer ${COMPUTER}"
else
    WANDB_NOTES="Pi0-FAST training on ${SINGLE_DATASET} with Computer ${COMPUTER}"
fi

# ============================================================================
# ENSURE LOG DIRECTORY EXISTS
# ============================================================================

mkdir -p "$REPO_ROOT/outputs/logs"

# ============================================================================
# TRAINING COMMAND
# ============================================================================

echo "=========================================="
echo "Pi0-FAST Training Configuration"
echo "=========================================="
echo "Computer:      $COMPUTER"
echo "Batch Size:    $BATCH_SIZE"
echo "Steps:         $STEPS"
echo "Action Chunk:  $CHUNK_SIZE"
echo "Job Name:      $JOB_NAME"
echo "Output Dir:    $OUTPUT_DIR"
echo "=========================================="

cd "$REPO_ROOT"

# Note: Pi0-FAST uses PaliGemma (SigLIP + Gemma).
# If you OOM on 3090Ti, try reducing resolution in input_features or batch_size=1
# The default policy.max_action_tokens is 256 

CMD=(python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="$DATASET_REPO_IDS" \
  --dataset.video_backend=pyav \
  --policy.type=pi0_fast \
  --policy.pretrained_path=lerobot/pi0_fast_base \
  --policy.repo_id="$REPO_ID" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.max_action_tokens=256 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.input_features='{
    "observation.images.top": {"shape": [3, 720, 1280], "type": "VISUAL"},
    "observation.images.left_wrist": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.images.right_wrist": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.state": {"shape": [15], "type": "STATE"},
    "observation.instruction": {"type": "LANGUAGE", "shape": [1]}
  }' \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --dataset.image_transforms.enable=false \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=true)

if [[ "$1" == "--no-daemon" ]]; then
    echo "Starting training in FOREGROUND..."
    "${CMD[@]}"
else
    echo "Starting training in BACKGROUND..."
    echo "Monitor progress: tail -f $LOG_FILE"
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    mkdir -p "$(dirname "$REPO_ROOT/outputs/logs/${JOB_NAME}.pid")"
    echo "$TRAIN_PID" > "$REPO_ROOT/outputs/logs/${JOB_NAME}.pid"
    echo "Training started with PID: $TRAIN_PID"
fi