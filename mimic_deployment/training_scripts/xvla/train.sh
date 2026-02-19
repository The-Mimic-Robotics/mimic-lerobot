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

DATASET_GROUP="${DATASET_GROUP:-}"
SINGLE_DATASET="${SINGLE_DATASET:-}"

# === POWER SETTINGS (RTX 3090) ===
BATCH_SIZE="${BATCH_SIZE:-16}"  # 24GB allows nice large batches
NUM_WORKERS="${NUM_WORKERS:-8}"
STEPS="${STEPS:-300000}" 
SAVE_FREQ="${SAVE_FREQ:-50000}" 
ACTION_STEPS="${ACTION_STEPS:-50}" 
CHUNK_SIZE="${CHUNK_SIZE:-50}"

# Resolve Dataset
if [ -n "$DATASET_GROUP" ]; then
    DATASET_REPO_IDS=$(python3 "$DATASET_RESOLVER" "$DATASET_GROUP" --format bash)
    DATASET_NAME_FOR_JOB="$DATASET_GROUP"
else
    DATASET_REPO_IDS="$SINGLE_DATASET"
    DATASET_NAME_FOR_JOB=$(echo "$SINGLE_DATASET" | sed 's|.*/||')
fi

DATASET_NAME_CLEAN=$(echo "$DATASET_NAME_FOR_JOB" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Auto-generate job name with TIMESTAMP
DATE_STR=$(date +"%Y%m%d_%H%M%S")
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}_${DATE_STR}"
fi
OUTPUT_DIR="$REPO_ROOT/outputs/train/${JOB_NAME}"
LOG_FILE="$REPO_ROOT/outputs/logs/${JOB_NAME}.log"
REPO_ID="Mimic-Robotics/${POLICY_TYPE}_${COMPUTER}_${DATASET_NAME_CLEAN}"

mkdir -p "$REPO_ROOT/outputs/logs"

echo "=========================================="
echo "X-VLA Training (RTX 3090 Mode)"
echo "Batch Size: $BATCH_SIZE"
echo "Dataset:    $DATASET_NAME_FOR_JOB"
echo "Log File:   $LOG_FILE"
echo "=========================================="

cd "$REPO_ROOT"

# === THE FIX ===
# 1. Removed explicit `input_features` (Let it load from pretrained config)
# 2. Added `rename_map`: Maps YOUR cameras to PRETRAINED slots.
#    - top        -> image (Primary)
#    - left_wrist -> image2 (Secondary)
#    - right_wrist -> empty_camera_0 (Tertiary slot)

CMD=(python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="$DATASET_REPO_IDS" \
  --dataset.video_backend=pyav \
  --policy.path="lerobot/xvla-base" \
  --policy.repo_id="$REPO_ID" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.action_mode=auto \
  --policy.num_image_views=2 \
  --policy.train_soft_prompts=true \
  --policy.train_policy_transformer=true \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.dtype=bfloat16 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.device=cuda \
  --dataset.image_transforms.enable=false \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=true \
  --rename_map='{
      "observation.images.left_wrist": "observation.images.image",
      "observation.images.right_wrist": "observation.images.image2"
  }' \
  --policy.input_features='{
    "observation.images.image": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.images.image2": {"shape": [3, 480, 640], "type": "VISUAL"},
    "observation.state": {"shape": [15], "type": "STATE"},
    "observation.instruction": {"shape": [1], "type": "LANGUAGE"}
  }')
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

if [[ "$1" == "--no-daemon" ]]; then
    echo "Starting in FOREGROUND..."
    "${CMD[@]}"
else
    echo "Starting in BACKGROUND..."
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    echo "Tail logs with: tail -f $LOG_FILE"
fi