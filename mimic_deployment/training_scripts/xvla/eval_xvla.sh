# Run lerobot-record for evaluation
lerobot-record \
  --robot.type=$ROBOT_TYPE \
  --robot.left_arm_port=$ROBOT_LEFT_ARM \
  --robot.right_arm_port=$ROBOT_RIGHT_ARM \
  --robot.base_port=$ROBOT_BASE \
  --robot.id=$ROBOT_ID \
  --robot.cameras="$CAMERAS" \
  --display_data=true \
  --dataset.repo_id=$EVAL_DATASET \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.single_task="$TASK" \
  --policy.path=$MODEL_PATH
