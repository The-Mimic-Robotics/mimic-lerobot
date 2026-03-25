#!/bin/bash
# X-VLA Training Script - RTX 3090 "Full Power" Edition
# 24GB VRAM Config: Batch 16 + Full VLM Finetuning

set -e

# Optimize memory allocation for 24GB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_MIXED_PRECISION="bf16"

POLICY_TYPE="xvla"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_RESOLVER="$REPO_ROOT/mimic_deployment/training_scripts/dataset_groups.py"
COMPUTER="${COMPUTER:-$(hostname)}"

XVLA_SPEED_MODE="${XVLA_SPEED_MODE:-default}"
if [[ "$XVLA_SPEED_MODE" == "smoke1k" ]]; then
  exec "$SCRIPT_DIR/train_speed_smoke_1k.sh" "$@"
fi
if [[ "$XVLA_SPEED_MODE" == "maxbatch" ]]; then
  exec "$SCRIPT_DIR/train_speed_max_batch_probe.sh" "$@"
fi

DATASET_GROUP="${DATASET_GROUP:-}"
SINGLE_DATASET="${SINGLE_DATASET:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
HF_PUSH_CHECKPOINTS="${HF_PUSH_CHECKPOINTS:-true}"
HF_CHECKPOINT_SYNC_INTERVAL="${HF_CHECKPOINT_SYNC_INTERVAL:-180}"
XVLA_POLICY_PATH="${XVLA_POLICY_PATH:-lerobot/xvla-base}"
XVLA_FREEZE_VISION_ENCODER="${XVLA_FREEZE_VISION_ENCODER:-false}"
XVLA_FREEZE_LANGUAGE_ENCODER="${XVLA_FREEZE_LANGUAGE_ENCODER:-false}"
XVLA_TRAIN_POLICY_TRANSFORMER="${XVLA_TRAIN_POLICY_TRANSFORMER:-true}"
XVLA_TRAIN_SOFT_PROMPTS="${XVLA_TRAIN_SOFT_PROMPTS:-true}"

# === POWER SETTINGS (RTX 3090) ===
BATCH_SIZE="${BATCH_SIZE:-9}"  # 24GB allows batch 9
NUM_WORKERS="${NUM_WORKERS:-8}"
STEPS="${STEPS:-300000}" 
SAVE_FREQ="${SAVE_FREQ:-50000}" 
ACTION_STEPS="${ACTION_STEPS:-32}" 
CHUNK_SIZE="${CHUNK_SIZE:-32}"

# Resolve Dataset
if [ -n "$DATASET_GROUP" ]; then
    DATASET_REPO_IDS=$(python3 "$DATASET_RESOLVER" "$DATASET_GROUP" --format bash)
    DATASET_NAME_FOR_JOB="$DATASET_GROUP"
else
    DATASET_REPO_IDS="$SINGLE_DATASET"
    DATASET_NAME_FOR_JOB=$(echo "$SINGLE_DATASET" | sed 's|.*/||')
fi

DATASET_NAME_CLEAN=$(echo "$DATASET_NAME_FOR_JOB" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
DATE_TAG=$(date +"%d_%b" | tr '[:upper:]' '[:lower:]')
TIME_TAG=$(date +"%H%M%S")
CKPT_TAG="lastckpt"
if [[ "$HF_PUSH_CHECKPOINTS" == "true" ]]; then
  CKPT_TAG="allckpt"
fi
BATCH_TAG="b${BATCH_SIZE}"
MODEL_BASENAME="${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}_${BATCH_TAG}_${DATE_TAG}_${CKPT_TAG}"
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}_${BATCH_TAG}_${DATE_TAG}_${TIME_TAG}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
OUTPUT_DIR="$OUTPUT_BASE/train/${JOB_NAME}"
LOG_FILE="$OUTPUT_BASE/logs/${JOB_NAME}.log"
REPO_ID="Mimic-Robotics/${MODEL_BASENAME}"

mkdir -p "$OUTPUT_BASE/logs" "$OUTPUT_BASE/train"

echo "=========================================="
echo "X-VLA Training (RTX 3090 Mode)"
echo "Batch Size: $BATCH_SIZE"
echo "Dataset:    $DATASET_NAME_FOR_JOB"
echo "Repo ID:    $REPO_ID"
echo "CKPT Sync:  $HF_PUSH_CHECKPOINTS (interval=${HF_CHECKPOINT_SYNC_INTERVAL}s)"
echo "Log File:   $LOG_FILE"
echo "=========================================="

cd "$REPO_ROOT"

LAUNCHER=(python)
MIG_MODE=""
if command -v nvidia-smi >/dev/null 2>&1; then
  MIG_MODE="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | head -n 1 | xargs || true)"
fi

if [[ "${LEROBOT_TORCHRUN_NPROC:-1}" =~ ^[0-9]+$ ]] && [ "${LEROBOT_TORCHRUN_NPROC:-1}" -gt 1 ]; then
  if [ "$MIG_MODE" = "Enabled" ]; then
    echo "Distributed launcher requested (${LEROBOT_TORCHRUN_NPROC}) but MIG is enabled; falling back to single-process launch to avoid NCCL duplicate-GPU errors."
  else
    LAUNCHER=(torchrun --standalone --nproc_per_node="${LEROBOT_TORCHRUN_NPROC}")
    echo "Distributed launcher: ${LAUNCHER[*]}"
  fi
fi

# === THE FIX ===
# 1. Removed explicit `input_features` (Let it load from pretrained config)
# 2. Added `rename_map`: Maps YOUR cameras to PRETRAINED slots.
#    - top        -> image (Primary)
#    - left_wrist -> image2 (Secondary)
#    - right_wrist -> empty_camera_0 (Tertiary slot)

CMD=("${LAUNCHER[@]}" src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="$DATASET_REPO_IDS" \
  --dataset.video_backend=pyav \
  --policy.path="$XVLA_POLICY_PATH" \
  --policy.repo_id="$REPO_ID" \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.action_mode=auto \
  --policy.max_action_dim=20 \
  --policy.num_image_views=3 \
  --policy.train_soft_prompts="$XVLA_TRAIN_SOFT_PROMPTS" \
  --policy.train_policy_transformer="$XVLA_TRAIN_POLICY_TRANSFORMER" \
  --policy.freeze_vision_encoder="$XVLA_FREEZE_VISION_ENCODER" \
  --policy.freeze_language_encoder="$XVLA_FREEZE_LANGUAGE_ENCODER" \
  --policy.dtype=bfloat16 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.device=cuda \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.random_order=true \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=true \
  --rename_map='{
    "observation.images.top": "observation.images.image",
      "observation.images.left_wrist": "observation.images.image2",
      "observation.images.right_wrist": "observation.images.image3"
  }' \
--policy.input_features='{
  "observation.images.image": {"shape": [3, 480, 640], "type": "VISUAL"},
  "observation.images.image2": {"shape": [3, 480, 640], "type": "VISUAL"},
  "observation.images.image3": {"shape": [3, 480, 640], "type": "VISUAL"},
  "observation.state": {"shape": [15], "type": "STATE"},
  "observation.instruction": {"shape": [1], "type": "LANGUAGE"}
}')


#   --policy.input_features='{
#     "observation.images.image": {"shape": [3, 480, 640], "type": "VISUAL"},
#     "observation.images.image2": {"shape": [3, 480, 640], "type": "VISUAL"},
#     "observation.state": {"shape": [15], "type": "STATE"},
#     "observation.instruction": {"shape": [1], "type": "LANGUAGE"}
#   }')
#   --rename_map='{
#       "observation.images.top": "observation.images.image",
#       "observation.images.left_wrist": "observation.images.image2",
#       "observation.images.right_wrist": "observation.images.empty_camera_0",
#       "observation.images.front": "observation.images.image"
#   }' \

#   --policy.input_features='{
#     "observation.images.image": {"shape": [3, 720, 1280], "type": "VISUAL"},
#     "observation.images.image2": {"shape": [3, 480, 640], "type": "VISUAL"},
#     "observation.images.empty_camera_0": {"shape": [3, 480, 640], "type": "VISUAL"},
#     "observation.state": {"shape": [15], "type": "STATE"},
#     "observation.instruction": {"shape": [1], "type": "LANGUAGE"}
#   }'
  
  
# Note: Added "front" mapping just in case your dataset uses that name instead of top. 
# It won't hurt if the key doesn't exist.

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
  echo "Starting in FOREGROUND..."
  run_with_checkpoint_sync
else
    echo "Starting in BACKGROUND..."
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    echo "Tail logs with: tail -f $LOG_FILE"
fi