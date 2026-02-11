#!/bin/bash
# SmolVLA Training Script - Smart Configuration with Auto Normalization
# Automatically configures training based on $COMPUTER environment variable and dataset groups

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# CONFIGURATION
# ============================================================================

POLICY_TYPE="smolvla"
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

# Training parameters with computer-specific defaults (VLA models need less batch size)
if [[ "$COMPUTER" == "odin" ]] || [[ "$COMPUTER" == "ODIN-IEEE" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-4}" # SmolVLA can handle a bit more than Pi0
    NUM_WORKERS="${NUM_WORKERS:-8}"
elif [[ "$COMPUTER" == "jupiter" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
elif [[ "$COMPUTER" == "mathias" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
else
    BATCH_SIZE="${BATCH_SIZE:-1}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
fi

STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
ACTION_STEPS="${ACTION_STEPS:-50}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"

# ============================================================================
# RESOLVE DATASET GROUP TO DATASET LIST OR USE SINGLE DATASET
# ============================================================================

if [ ! -f "$DATASET_RESOLVER" ]; then
    echo "Error: Dataset resolver not found at $DATASET_RESOLVER"
    exit 1
fi

# Validate: must have either DATASET_GROUP or SINGLE_DATASET
if [ -z "$DATASET_GROUP" ] && [ -z "$SINGLE_DATASET" ]; then
    echo "Error: Either DATASET_GROUP or SINGLE_DATASET must be provided"
    exit 1
fi

if [ -n "$DATASET_GROUP" ] && [ -n "$SINGLE_DATASET" ]; then
    echo "Error: Cannot specify both DATASET_GROUP and SINGLE_DATASET"
    exit 1
fi

# Resolve datasets
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

# Sanitize dataset name for use in paths
DATASET_NAME_CLEAN=$(echo "$DATASET_NAME_FOR_JOB" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

# Auto-generate job name and output directory
# Only generate a new name if one wasn't passed from the manager
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"
fi
OUTPUT_DIR="$REPO_ROOT/outputs/train/${JOB_NAME}"
LOG_FILE="$REPO_ROOT/outputs/logs/${JOB_NAME}.log"

# Auto-generate repo ID for Hugging Face Hub
REPO_ID="Mimic-Robotics/${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"

# WandB notes
if [ -n "$DATASET_GROUP" ]; then
    WANDB_NOTES="Multi-dataset SmolVLA training on ${DATASET_GROUP} with Computer ${COMPUTER} and Batch ${BATCH_SIZE}"
else
    WANDB_NOTES="SmolVLA training on ${SINGLE_DATASET} with Computer ${COMPUTER} and Batch ${BATCH_SIZE}"
fi

# ============================================================================
# ENSURE LOG DIRECTORY EXISTS
# ============================================================================

mkdir -p "$REPO_ROOT/outputs/logs"

# ============================================================================
# TRAINING COMMAND
# ============================================================================

echo "=========================================="
echo "SmolVLA Training Configuration"
echo "=========================================="
echo "Computer:      $COMPUTER"
if [ -n "$DATASET_GROUP" ]; then
    echo "Dataset Group: $DATASET_GROUP"
else
    echo "Single Dataset: $SINGLE_DATASET"
fi
echo "Batch Size:    $BATCH_SIZE"
echo "Num Workers:   $NUM_WORKERS"
echo "Steps:         $STEPS"
echo "Job Name:      $JOB_NAME"
echo "Output Dir:    $OUTPUT_DIR"
echo "Repo ID:       $REPO_ID"
echo "Log File:      $LOG_FILE"
echo "=========================================="
echo ""
echo "Note: SmolVLA automatically handles normalization"
echo ""

cd "$REPO_ROOT"

# ============================================================================
# EXECUTION LOGIC (FOREGROUND VS BACKGROUND)
# ============================================================================

# common arguments for both modes
CMD=(python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="$DATASET_REPO_IDS" \
--dataset.video_backend=pyav \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.repo_id="$REPO_ID" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --policy.chunk_size="$CHUNK_SIZE" \
#   --gradient_checkpointing=true \
#   --dtype=bfloat16 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
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
    # --- FOREGROUND MODE ---
    echo "Starting training in FOREGROUND..."
    echo "Output will be shown directly in terminal."
    echo ""

    # Run command directly
    "${CMD[@]}"

else
    # --- BACKGROUND MODE ---
    echo "Starting training in BACKGROUND..."
    echo "Monitor progress: tail -f $LOG_FILE"
    echo ""

    # Run with nohup
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    # Create the log directory if it doesn't exist to ensure PID file can be written
    mkdir -p "$(dirname "$REPO_ROOT/outputs/logs/${JOB_NAME}.pid")"
    echo "$TRAIN_PID" > "$REPO_ROOT/outputs/logs/${JOB_NAME}.pid"

    echo "Training started with PID: $TRAIN_PID"
    echo "Log file: $LOG_FILE"
    echo "PID file: $REPO_ROOT/outputs/logs/${JOB_NAME}.pid"
    echo ""
    echo "To stop training:"
    echo "  kill $TRAIN_PID"
fi
