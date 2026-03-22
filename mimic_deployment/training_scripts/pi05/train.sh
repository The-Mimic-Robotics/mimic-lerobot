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

PI05_SPEED_MODE="${PI05_SPEED_MODE:-default}"
if [[ "$PI05_SPEED_MODE" == "maxbatch" ]]; then
    exec "$SCRIPT_DIR/train_speed_max_batch_probe.sh" "$@"
fi

# ============================================================================
# PARAMETERS (Override via command line arguments)
# ============================================================================

DATASET_GROUP="${DATASET_GROUP:-}"
SINGLE_DATASET="${SINGLE_DATASET:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

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
SAVE_FREQ="${SAVE_FREQ:-20000}"
ACTION_STEPS="${ACTION_STEPS:-32}" 
CHUNK_SIZE="${CHUNK_SIZE:-32}"
PI05_COMPILE_MODEL="${PI05_COMPILE_MODEL:-false}"
PI05_COMPILE_MODE="${PI05_COMPILE_MODE:-max-autotune}"
PI05_USE_PEFT="${PI05_USE_PEFT:-false}"
PI05_PEFT_TYPE="${PI05_PEFT_TYPE:-LORA}"
PI05_LORA_R="${PI05_LORA_R:-32}"
PI05_LORA_ALPHA="${PI05_LORA_ALPHA:-64}"
PI05_LORA_DROPOUT="${PI05_LORA_DROPOUT:-0.05}"
PI05_FREEZE_VISION_ENCODER="${PI05_FREEZE_VISION_ENCODER:-false}"
PI05_TRAIN_EXPERT_ONLY="${PI05_TRAIN_EXPERT_ONLY:-true}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-false}"

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

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
OUTPUT_DIR="$OUTPUT_BASE/train/${JOB_NAME}"
LOG_FILE="$OUTPUT_BASE/logs/${JOB_NAME}.log"
REPO_ID="Mimic-Robotics/${JOB_NAME}"

mkdir -p "$OUTPUT_BASE/logs" "$OUTPUT_BASE/train"

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
echo "Use PEFT:      $PI05_USE_PEFT"
echo "Freeze Vision: $PI05_FREEZE_VISION_ENCODER"
echo "Expert Only:   $PI05_TRAIN_EXPERT_ONLY"
echo "W&B Artifact:  $WANDB_DISABLE_ARTIFACT"
echo "Compile Model: $PI05_COMPILE_MODEL ($PI05_COMPILE_MODE)"
echo "Log File:      $LOG_FILE"
echo "=========================================="

cd "$REPO_ROOT"
CMD=(python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="$DATASET_REPO_IDS" \
  --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id="$REPO_ID" \
    --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --policy.chunk_size="$CHUNK_SIZE" \
    --policy.scheduler_decay_steps="$STEPS" \
  --policy.use_peft=false \
  --policy.train_expert_only="$PI05_TRAIN_EXPERT_ONLY" \
    --policy.freeze_vision_encoder="$PI05_FREEZE_VISION_ENCODER" \
    --policy.gradient_checkpointing=true \
    --policy.compile_model="$PI05_COMPILE_MODEL" \
    --policy.compile_mode="$PI05_COMPILE_MODE" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --dataset.image_transforms.enable=false \
    --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=true \
  --wandb.disable_artifact="$WANDB_DISABLE_ARTIFACT")

if [[ "$PI05_USE_PEFT" == "true" ]]; then
  CMD+=(
    --peft.method_type="$PI05_PEFT_TYPE"
    --peft.r="$PI05_LORA_R"
  )
fi

if [[ "${1:-}" == "--no-daemon" ]]; then
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