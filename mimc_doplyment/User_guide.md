
#TODO set custome ID to usb port

Verfify all the arm port using 

ls /dev/ttyACM*

then plug the base 
check 
 ls /dev/ttyUSB0


test : 


lerobot-teleoperate \
  --robot.type=mimic_follower \
  --robot.left_arm_port=/dev/ttyACM3 \
  --robot.right_arm_port=/dev/ttyACM2 \
  --robot.base_port=/dev/ttyUSB0 \
  --robot.id=mimic_follower \
  --robot.cameras='{}' \
  --teleop.type=mimic_leader \
  --teleop.left_arm_port=/dev/ttyACM0 \
  --teleop.right_arm_port=/dev/ttyACM1 \
  --teleop.base_control_mode=keyboard \
  --teleop.id=mimic_leader \
  --display_data=true


  add camera

  verify camera port 
  ls /dev/video*
  Test that the camera are good 

  python mimc_doplyment/screenshot/test_camera.py

  Right_wrist camera -> video 0
  left_wrist camera -> video 2
  zed camera -> video 4


  the Zed camera should look double .

  the zed camera should look quite bad, because no filtering is applied 
  better for models

  if not unplug replug or change in the follwoing commande line




teleop with camera 


lerobot-teleoperate \
  --robot.type=mimic_follower \
  --robot.left_arm_port=/dev/ttyACM3 \
  --robot.right_arm_port=/dev/ttyACM2 \
  --robot.base_port=/dev/ttyUSB0 \
  --robot.id=mimic_follower \
  --robot.cameras='{
    "right_wrist": {
      "type": "opencv",
      "index_or_path": 0,
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "left_wrist": {
      "type": "opencv",
      "index_or_path": 2,
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "head": {
      "type": "zed_camera",
      "index_or_path": 4,
      "width": 1280,
      "height": 720,
      "fps": 30
    }
  }' \
  --teleop.type=mimic_leader \
  --teleop.left_arm_port=/dev/ttyACM0 \
  --teleop.right_arm_port=/dev/ttyACM1 \
  --teleop.base_control_mode=keyboard \
  --teleop.id=mimic_leader \
  --display_data=true


  record

lerobot-record \
  --robot.type=mimic_follower \
  --robot.left_arm_port=/dev/ttyACM3 \
  --robot.right_arm_port=/dev/ttyACM2 \
  --robot.base_port=/dev/ttyUSB0 \
  --robot.id=mimic_follower_v1 \
  --robot.cameras='{}' \
  --teleop.type=mimic_leader \
  --teleop.left_arm_port=/dev/ttyACM0 \
  --teleop.right_arm_port=/dev/ttyACM1 \
  --teleop.base_control_mode=keyboard \
  --teleop.id=mimic_leader_v1 \
  --dataset.repo_id=<YOUR_USERNAME>/mimic_test_dataset \
  --dataset.single_task="Move forward and wave hands" \
  --dataset.num_episodes=2 \
  --display_data=true



 it should mimic the base mouvement too right? only the input or the actually recorded mouvement?