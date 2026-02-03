#!/bin/bash
cd ~/mimic-lerobot
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# All datasets with resize to 480x640 (smallest common size)
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id='[Mimic-Robotics/mimic_displacement_to_handover_blue_block_with_new_hands_v3,Mimic-Robotics/mimic_displacement_to_handover_blue_block_with_new_hands_v2,Mimic-Robotics/mimic_displacement_to_handover_blue_block_v8,Mimic-Robotics/mimic_displacement_to_handover_blue_block_v7,Mimic-Robotics/mimic_displacement_to_handover_blue_block_v6,Mimic-Robotics/mimic_displacement_to_handover_blue_block_v2,neryotw/bimanual_blue_block_handover_1_drift,neryotw/bimanual_blue_block_handover_2_drift,neryotw/bimanual_blue_block_handover_3_drift,neryotw/bimanual_blue_block_handover_4_drift,neryotw/bimanual_blue_block_handover_5_drift,neryotw/bimanual_blue_block_handover_6_drift,neryotw/bimanual_blue_block_handover_7_drift,neryotw/bimanual_blue_block_handover_14_drift,neryotw/bimanual_blue_block_handover_15_drift,neryotw/bimanual_blue_block_handover_16_drift,neryotw/bimanual_blue_block_handover_17_drift,neryotw/bimanual_blue_block_handover_18_drift,neryotw/bimanual_blue_block_handover_19_drift,neryotw/bimanual_blue_block_handover_20_drift,neryotw/bimanual_blue_block_handover_21_drift,neryotw/bimanual_blue_block_handover_22_drift,neryotw/bimanual_blue_block_handover_23_drift,neryotw/bimanual_blue_block_handover_24_drift,neryotw/bimanual_blue_block_handover_25_drift,neryotw/bimanual_blue_block_handover_26_drift,neryotw/mimic_mobile_bimanual_drift_v2_drift]' \
    --dataset.image_transforms.resize='[480, 640]' \
    --policy.path=lerobot/xvla-base \
    --policy.repo_id=Mimic-Robotics/xvla_bimanual_handover \
    --policy.dtype=bfloat16 \
    --policy.action_mode=auto \
    --policy.max_action_dim=20 \
    --policy.freeze_vision_encoder=false \
    --policy.freeze_language_encoder=false \
    --policy.train_policy_transformer=true \
    --policy.train_soft_prompts=true \
    --batch_size=4 \
    --steps=50000 \
    --save_freq=5000 \
    --output_dir=./outputs/xvla_all_datasets \
    --policy.num_image_views=4 \
    --rename_map='{"observation.images.top": "", "observation.images.front": "", "observation.images.left_wrist": ""}'
