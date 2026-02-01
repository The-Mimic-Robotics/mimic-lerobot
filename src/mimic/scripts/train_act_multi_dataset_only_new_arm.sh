#!/bin/bash
# ACT Multi-Dataset Training on Odin
# Includes: drift_v2 + all converted bimanual mobile datasets

COMPUTER="odin"
POLICY_TYPE="act"

# NOTE: Update this list with your actual converted dataset repo IDs
# Format: "mimic-robotics/mimic_mobile_bimanual_DATASET_NAME"
# The drift_v2 dataset is already included below

nohup lerobot-train \
  --dataset.repo_id='["Mimic-Robotics/mimic_displacement_to_handover_blue_block_with_new_hands_v2", "Mimic-Robotics/mimic_displacement_to_handover_blue_block_with_new_hands_v3"]' \
  --policy.type=$POLICY_TYPE \
  --output_dir=outputs/train/${POLICY_TYPE}_${COMPUTER}_only_new_arms1 \
  --job_name=${POLICY_TYPE}_${COMPUTER}_only_new_arms1 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.notes="Multi-dataset ACT training on new arms only, 40 episodes" \
  --policy.repo_id="Mimic-Robotics/${POLICY_TYPE}_${COMPUTER}_only_new_arms1" \
  --batch_size=64 \
  --num_workers=14 \
  --steps=100000 \
  --eval_freq=10000 \
  --log_freq=250 \
  --save_checkpoint=true \
  > outputs/logs/${POLICY_TYPE}_${COMPUTER}_only_new_arms1.log 2>&1 &

echo "Training started in background. Check logs with:"
echo "tail -f outputs/logs/${POLICY_TYPE}_${COMPUTER}_only_new_arms1.log"