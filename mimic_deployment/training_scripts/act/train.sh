#!/bin/bash
# ACT Training Script - Smart Configuration
# Automatically configures training based on $COMPUTER environment variable and dataset groups

set -e

# ACT typically uses fp16 (unlike SmolVLA which prefers bf16)
export ACCELERATE_MIXED_PRECISION="fp16"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# CONFIGURATION
# ============================================================================

POLICY_TYPE="act"
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

# Training parameters with computer-specific defaults
# ACT is lighter than VLA, so we can usually afford higher batch sizes
if [[ "$COMPUTER" == "odin" ]] || [[ "$COMPUTER" == "ODIN-IEEE" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-8}"
    NUM_WORKERS="${NUM_WORKERS:-12}"
elif [[ "$COMPUTER" == "jupiter" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-6}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
elif [[ "$COMPUTER" == "mathias" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-6}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
else
    BATCH_SIZE="${BATCH_SIZE:-8}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
fi

STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}" # Save more often to catch early convergence
ACTION_STEPS="${ACTION_STEPS:-100}" # ACT defaults to 100
CHUNK_SIZE="${CHUNK_SIZE:-100}"     # ACT defaults to 100

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
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"
fi
OUTPUT_DIR="$REPO_ROOT/outputs/train/${JOB_NAME}"
LOG_FILE="$REPO_ROOT/outputs/logs/${JOB_NAME}.log"

# Auto-generate repo ID for Hugging Face Hub
REPO_ID="Mimic-Robotics/${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"

# WandB notes
if [ -n "$DATASET_GROUP" ]; then
    WANDB_NOTES="Multi-dataset ACT training on ${DATASET_GROUP} with Computer ${COMPUTER} and Batch ${BATCH_SIZE}"
else
    WANDB_NOTES="ACT training on ${SINGLE_DATASET} with Computer ${COMPUTER} and Batch ${BATCH_SIZE}"
fi

# ============================================================================
# ENSURE LOG DIRECTORY EXISTS
# ============================================================================

mkdir -p "$REPO_ROOT/outputs/logs"

# ============================================================================
# TRAINING COMMAND
# ============================================================================

echo "=========================================="
echo "ACT Training Configuration"
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

cd "$REPO_ROOT"

# ============================================================================
# EXECUTION LOGIC
# ============================================================================

# Note: We do NOT use 'observation.instruction' for standard ACT
CMD=(python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="$DATASET_REPO_IDS" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.repo_id="$REPO_ID" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.dim_model=512 \
  --policy.input_features='{
    "observation.images.top": {"shape": [3, 720, 1280], "type": "VISUAL"},
    "observation.images.left_wrist": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.images.right_wrist": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.state": {"shape": [15], "type": "STATE"}
  }' \
  --optimizer.lr=2e-5 \
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