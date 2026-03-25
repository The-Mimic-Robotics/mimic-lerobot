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
HF_PUSH_CHECKPOINTS="${HF_PUSH_CHECKPOINTS:-true}"
HF_CHECKPOINT_SYNC_INTERVAL="${HF_CHECKPOINT_SYNC_INTERVAL:-180}"

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
PI05_COMPILE_MODEL="${PI05_COMPILE_MODEL:-true}"
PI05_COMPILE_MODE="${PI05_COMPILE_MODE:-max-autotune}"
PI05_PRETRAINED_PATH="${PI05_PRETRAINED_PATH:-lerobot/pi05_base}"
PI05_GRADIENT_CHECKPOINTING="${PI05_GRADIENT_CHECKPOINTING:-true}"
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
echo "Freeze Vision: $PI05_FREEZE_VISION_ENCODER"
echo "Expert Only:   $PI05_TRAIN_EXPERT_ONLY"
echo "W&B Artifact:  $WANDB_DISABLE_ARTIFACT"
echo "Compile Model: $PI05_COMPILE_MODEL ($PI05_COMPILE_MODE)"
echo "Gradient Ckpt: $PI05_GRADIENT_CHECKPOINTING"
echo "Pretrained Path: $PI05_PRETRAINED_PATH"
echo "CKPT Sync:     $HF_PUSH_CHECKPOINTS (interval=${HF_CHECKPOINT_SYNC_INTERVAL}s)"
echo "Log File:      $LOG_FILE"
echo "=========================================="

cd "$REPO_ROOT"
CMD=(python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="$DATASET_REPO_IDS" \
  --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path="$PI05_PRETRAINED_PATH" \
  --policy.repo_id="$REPO_ID" \
    --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --policy.chunk_size="$CHUNK_SIZE" \
    --policy.scheduler_decay_steps="$STEPS" \
  --policy.use_peft=false \
  --policy.train_expert_only="$PI05_TRAIN_EXPERT_ONLY" \
    --policy.freeze_vision_encoder="$PI05_FREEZE_VISION_ENCODER" \
    --policy.gradient_checkpointing="$PI05_GRADIENT_CHECKPOINTING" \
    --policy.compile_model="$PI05_COMPILE_MODEL" \
    --policy.compile_mode="$PI05_COMPILE_MODE" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --dataset.image_transforms.enable=true \
    --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=true \
  --wandb.disable_artifact="$WANDB_DISABLE_ARTIFACT")

sync_checkpoints_to_hf() {
  local sync_db="$OUTPUT_DIR/.hf_uploaded_checkpoints"

  if [[ "$HF_PUSH_CHECKPOINTS" != "true" ]]; then
    return 0
  fi

  if [ ! -d "$OUTPUT_DIR" ]; then
    return 0
  fi
  touch "$sync_db"

  local hf_cmd=""
  if command -v hf >/dev/null 2>&1; then
    hf_cmd="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    hf_cmd="huggingface-cli"
  else
    echo "[ckpt-sync] neither hf nor huggingface-cli is available; skipping checkpoint sync."
    return 0
  fi

  while IFS= read -r ckpt_dir; do
    [ -d "$ckpt_dir" ] || continue
    local ckpt_name
    ckpt_name="$(basename "$ckpt_dir")"

    if grep -Fxq "$ckpt_name" "$sync_db"; then
      continue
    fi

    echo "[ckpt-sync] uploading checkpoint: $ckpt_name"
    if "$hf_cmd" upload "$REPO_ID" "$ckpt_dir" "checkpoints/$ckpt_name" --repo-type model --commit-message "Add checkpoint $ckpt_name"
    then
      echo "$ckpt_name" >> "$sync_db"
      echo "[ckpt-sync] uploaded: $ckpt_name"
    else
      echo "[ckpt-sync] upload failed for: $ckpt_name"
    fi
  done < <(find "$OUTPUT_DIR" -maxdepth 2 -type d \( -path "$OUTPUT_DIR/checkpoints/*" -o -path "$OUTPUT_DIR/checkpoint-*" \) | sort)
}

run_with_checkpoint_sync() {
  "${CMD[@]}" &
  local train_pid=$!

  (
    set +e
    while kill -0 "$train_pid" 2>/dev/null; do
      sync_checkpoints_to_hf
      sleep "$HF_CHECKPOINT_SYNC_INTERVAL"
    done
    sync_checkpoints_to_hf
  ) &
  local sync_pid=$!

  wait "$train_pid"
  local train_status=$?
  wait "$sync_pid" || true
  return "$train_status"
}

if [[ "${1:-}" == "--no-daemon" ]]; then
    echo "Starting training in FOREGROUND..."
    run_with_checkpoint_sync
else
    echo "Starting training in BACKGROUND..."
    mkdir -p "$(dirname "$LOG_FILE")"
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$REPO_ROOT/outputs/logs/${JOB_NAME}.pid"
    echo "Training started with PID: $TRAIN_PID"
    echo "Monitor with: tail -f $LOG_FILE"
fi