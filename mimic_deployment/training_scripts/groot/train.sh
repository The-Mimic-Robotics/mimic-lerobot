#!/bin/bash
# Groot Training Script - Smart Configuration with Accelerate and Auto GPU Detection
# Automatically configures training based on $COMPUTER environment variable and dataset groups

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

POLICY_TYPE="groot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_RESOLVER="$REPO_ROOT/mimic_deployment/training_scripts/dataset_groups.py"

# Get computer name from environment or hostname
COMPUTER="${COMPUTER:-$(hostname)}"

# ============================================================================
# AUTO-DETECT GPU COUNT
# ============================================================================

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $NUM_GPUS GPU(s)"
else
    echo "Warning: nvidia-smi not found, defaulting to 1 GPU"
    NUM_GPUS=1
fi

# ============================================================================
# PARAMETERS (Override via command line arguments)
# ============================================================================

# Dataset group (required)
DATASET_GROUP="${DATASET_GROUP:-}"
SINGLE_DATASET="${SINGLE_DATASET:-}"

# Training parameters with computer-specific defaults
if [[ "$COMPUTER" == "odin" ]] || [[ "$COMPUTER" == "ODIN-IEEE" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-4}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
elif [[ "$COMPUTER" == "jupiter" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
elif [[ "$COMPUTER" == "mathias" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
else
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
fi

STEPS="${STEPS:-15000}"
SAVE_FREQ="${SAVE_FREQ:-3000}"
LOG_FREQ="${LOG_FREQ:-100}"

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
JOB_NAME="${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"
OUTPUT_DIR="$REPO_ROOT/outputs/train/${JOB_NAME}"
LOG_FILE="$REPO_ROOT/outputs/logs/${JOB_NAME}.log"

# Auto-generate repo ID for Hugging Face Hub
REPO_ID="Mimic-Robotics/${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"

# WandB notes
if [ -n "$DATASET_GROUP" ]; then
    WANDB_NOTES="Multi-dataset Groot training on ${DATASET_GROUP} with Computer ${COMPUTER}, GPUs ${NUM_GPUS}, and Batch ${BATCH_SIZE}"
else
    WANDB_NOTES="Groot training on ${SINGLE_DATASET} with Computer ${COMPUTER}, GPUs ${NUM_GPUS}, and Batch ${BATCH_SIZE}"
fi

# ============================================================================
# ENSURE LOG DIRECTORY EXISTS
# ============================================================================

mkdir -p "$REPO_ROOT/outputs/logs"

# ============================================================================
# TRAINING COMMAND (with accelerate for multi-GPU)
# ============================================================================

echo "=========================================="
echo "Groot Training Configuration"
echo "=========================================="
echo "Computer:      $COMPUTER"
if [ -n "$DATASET_GROUP" ]; then
    echo "Dataset Group: $DATASET_GROUP"
else
    echo "Single Dataset: $SINGLE_DATASET"
fi
echo "Num GPUs:      $NUM_GPUS"
echo "Batch Size:    $BATCH_SIZE"
echo "Num Workers:   $NUM_WORKERS"
echo "Steps:         $STEPS"
echo "Job Name:      $JOB_NAME"
echo "Output Dir:    $OUTPUT_DIR"
echo "Repo ID:       $REPO_ID"
echo "Log File:      $LOG_FILE"
echo "=========================================="
echo ""
echo "Starting training in background with accelerate..."
echo "Monitor progress: tail -f $LOG_FILE"
echo ""

cd "$REPO_ROOT"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training
    nohup accelerate launch \
      --multi_gpu \
      --num_processes="$NUM_GPUS" \
      $(which lerobot-train) \
      --dataset.repo_id="$DATASET_REPO_IDS" \
      --policy.type="$POLICY_TYPE" \
      --policy.repo_id="$REPO_ID" \
      --policy.tune_diffusion_model=false \
      --policy.device=cuda \
      --batch_size="$BATCH_SIZE" \
      --num_workers="$NUM_WORKERS" \
      --steps="$STEPS" \
      --save_checkpoint=true \
      --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ" \
      --output_dir="$OUTPUT_DIR" \
      --job_name="$JOB_NAME" \
      --wandb.enable=true \
      --wandb.disable_artifact=true \
      --wandb.notes="$WANDB_NOTES" \
      > "$LOG_FILE" 2>&1 &
else
    # Single GPU training
    nohup lerobot-train \
      --dataset.repo_id="$DATASET_REPO_IDS" \
      --policy.type="$POLICY_TYPE" \
      --policy.repo_id="$REPO_ID" \
      --policy.tune_diffusion_model=false \
      --policy.device=cuda \
      --batch_size="$BATCH_SIZE" \
      --num_workers="$NUM_WORKERS" \
      --steps="$STEPS" \
      --save_checkpoint=true \
      --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ" \
      --output_dir="$OUTPUT_DIR" \
      --job_name="$JOB_NAME" \
      --wandb.enable=true \
      --wandb.disable_artifact=true \
      --wandb.notes="$WANDB_NOTES" \
      > "$LOG_FILE" 2>&1 &
fi

TRAIN_PID=$!
echo "$TRAIN_PID" > "$REPO_ROOT/outputs/logs/${JOB_NAME}.pid"

echo "Training started with PID: $TRAIN_PID"
echo "Log file: $LOG_FILE"
echo "PID file: $REPO_ROOT/outputs/logs/${JOB_NAME}.pid"
