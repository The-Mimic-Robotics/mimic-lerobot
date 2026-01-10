#!/bin/bash

# XVLA Training for Mobile Bimanual Robot
# Dataset: Mimic-Robotics/mimic_mobile_bimanual_drift_v2
# Policy: Mobile Bimanual SO101 with XVLA

lerobot-train \
  --dataset.repo_id=Mimic-Robotics/mimic_mobile_bimanual_drift_v2 \
  --dataset.revision=main \
  --output_dir=./outputs/xvla_mobile_bimanual_$(date +%Y%m%d_%H%M%S) \
  --job_name=xvla_mobile_bimanual_drift_v2 \
  --policy.path="lerobot/xvla-base" \
  --policy.action_mode=auto \
  --policy.max_action_dim=20 \
  --policy.dtype=bfloat16 \
  --policy.repo_id="Mimic-Robotics/xvla-mobile-bimanual-drift" \
  --steps=10000 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --save_freq=1000 \
  --save_checkpoint=true \
  --eval_freq=500 \
  --log_freq=100 \
  --wandb.enable=true \
  --wandb.project="xvla-mobile-bimanual" \
  --wandb.notes="Training XVLA on mobile bimanual drift dataset v2" \
  --rename_map='{"observation.images.top": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.empty_camera_0"}'
