#!/bin/bash

# ACT Multi-Dataset Training on Odin
# Includes: drift_v2 + all converted bimanual mobile datasets

COMPUTER="odin"
POLICY_TYPE="act"

# NOTE: Update this list with your actual converted dataset repo IDs
# Format: "mimic-robotics/mimic_mobile_bimanual_DATASET_NAME"
# The drift_v2 dataset is already included below

nohup lerobot-train \
  --dataset.repo_id='["Mimic-Robotics/mimic_mobile_bimanual_drift_v2", "Mimic-Robotics/mobile_bimanual_blue_block_handover_1", "Mimic-Robotics/mobile_bimanual_blue_block_handover_2", "Mimic-Robotics/mobile_bimanual_blue_block_handover_3", "Mimic-Robotics/mobile_bimanual_blue_block_handover_4", "Mimic-Robotics/mobile_bimanual_blue_block_handover_5", "Mimic-Robotics/mobile_bimanual_blue_block_handover_6", "Mimic-Robotics/mobile_bimanual_blue_block_handover_7", "Mimic-Robotics/mobile_bimanual_blue_block_handover_14", "Mimic-Robotics/mobile_bimanual_blue_block_handover_15", "Mimic-Robotics/mobile_bimanual_blue_block_handover_16", "Mimic-Robotics/mobile_bimanual_blue_block_handover_17", "Mimic-Robotics/mobile_bimanual_blue_block_handover_18", "Mimic-Robotics/mobile_bimanual_blue_block_handover_19", "Mimic-Robotics/mobile_bimanual_blue_block_handover_20", "Mimic-Robotics/mobile_bimanual_blue_block_handover_21", "Mimic-Robotics/mobile_bimanual_blue_block_handover_22", "Mimic-Robotics/mobile_bimanual_blue_block_handover_23", "Mimic-Robotics/mobile_bimanual_blue_block_handover_24", "Mimic-Robotics/mobile_bimanual_blue_block_handover_25", "Mimic-Robotics/mobile_bimanual_blue_block_handover_26"]' \
  --policy.type=$POLICY_TYPE \
  --output_dir=outputs/train/${POLICY_TYPE}_${COMPUTER}_Mobile_Bimanual_Multi_FromScratch \
  --job_name=${POLICY_TYPE}_${COMPUTER}_Mobile_Bimanual_Multi_FromScratch \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.notes="Multi-dataset ACT training on drift_v2 + 20 converted mobile bimanual handover datasets - $POLICY_TYPE on $COMPUTER - From Scratch" \
  --policy.repo_id="Mimic-Robotics/${POLICY_TYPE}_${COMPUTER}_mobile_bimanual_combined" \
  --batch_size=32 \
  --num_workers=8 \
  --steps=100000 \
  --eval_freq=10000 \
  --log_freq=250 \
  --save_checkpoint=true \
  > outputs/logs/${POLICY_TYPE}_${COMPUTER}_Mobile_Bimanual_Multi_FromScratch.log 2>&1 &

echo "Training started in background. Check logs with:"
echo "tail -f outputs/logs/${POLICY_TYPE}_${COMPUTER}_Mobile_Bimanual_Multi_FromScratch.log"