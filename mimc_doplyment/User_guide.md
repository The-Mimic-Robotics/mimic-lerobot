
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

  note : "If you see lag: It is usually because the "Image Writer" threads are fighting for CPU. You can tune this flag if needed:

    --dataset.num_image_writer_processes=1 (Spawns a separate process for saving images, keeping the main loop fast).

    --dataset.video=true (This is on by default, but double-check it stays true; encoding video is often faster than writing thousands of individual PNGs)."

note check user "```bash
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
```"

lerobot-record \
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
  --dataset.repo_id=Mimic-Robotics/mimic_zed_data_test \
  --dataset.single_task="Testing ZED camera integration" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=60 \
   --dataset.reset_time_s= 10 \
  --dataset.num_image_writer_processes=1 \
  --dataset.video=true \
  --dataset.fps=30 \
  --display_data=true


Replay test : 

lerobot-replay \
  --robot.type=mimic_follower \
  --robot.left_arm_port=/dev/ttyACM3 \
  --robot.right_arm_port=/dev/ttyACM2 \
  --robot.base_port=/dev/ttyUSB0 \
  --robot.id=mimic_follower \
  --dataset.repo_id=Mimic-Robotics/mimic_zed_data_test \
  --dataset.episode=0 \
  --dataset.fps=30




