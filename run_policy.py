#!/usr/bin/env python3
"""Run trained ACT policy on Mimic robot."""

import time
import torch
import torch.nn.functional as F
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.mimic_follower.mimic_follower import MimicFollower


def main():
    # Load policy from HuggingFace
    print("Loading policy from neryotw/act_bimanual_drift...")
    policy = ACTPolicy.from_pretrained("neryotw/act_bimanual_drift")
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    print(f"Policy loaded on {device}")

    # Initialize robot
    print("Initializing robot...")
    from lerobot.robots.mimic_follower.config_mimic_follower import MimicFollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.zed_camera import ZedCameraConfig

    config = MimicFollowerConfig(
        id="mimic_follower",
        calibration_dir=Path("~/.cache/huggingface/lerobot/calibration/robots/so100_follower").expanduser(),
        left_arm_port="/dev/arm_left_follower",
        right_arm_port="/dev/arm_right_follower",
        base_port="/dev/mecanum_base",
        cameras={
            "wrist_right": OpenCVCameraConfig(index_or_path="/dev/camera_right_wrist", width=640, height=480, fps=30, fourcc="MJPG"),
            "wrist_left": OpenCVCameraConfig(index_or_path="/dev/camera_left_wrist", width=640, height=480, fps=30, fourcc="MJPG"),
            "top": ZedCameraConfig(index_or_path="/dev/camera_zed", width=1280, height=720, fps=30),
            "front": OpenCVCameraConfig(index_or_path="/dev/camera_front", width=1280, height=720, fps=30, fourcc="MJPG"),
        }
    )
    robot = MimicFollower(config)
    robot.connect(calibrate=False)
    print("Robot connected!")

    # Debug: check camera data
    print("\nChecking cameras...")
    test_obs = robot.get_observation()
    for key in ["wrist_right", "wrist_left", "top", "front"]:
        if key in test_obs:
            img = test_obs[key]
            print(f"  {key}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
        else:
            print(f"  {key}: NOT FOUND")

    # Debug: check state keys from robot
    print("\nState keys from robot:")
    state_keys = [k for k in test_obs.keys() if not isinstance(test_obs[k], (list, tuple)) or not hasattr(test_obs[k], 'shape')]
    state_keys = [k for k in test_obs.keys() if k not in ["wrist_right", "wrist_left", "top", "front"]]
    for key in sorted(state_keys):
        print(f"  {key}: {test_obs[key]}")

    # Debug: check what policy expects
    print("\nPolicy config:")
    for attr in ['input_features', 'output_features', 'state_feature', 'action_feature']:
        if hasattr(policy.config, attr):
            print(f"  {attr}: {getattr(policy.config, attr)}")

    print("\nRunning policy at 30 FPS. Press Ctrl+C to stop.\n")

    try:
        while True:
            start = time.time()

            # Get observation
            obs = robot.get_observation()

            # Build state vector (12D arm positions + 3D base = 15D)
            state = []
            for key in ["left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
                        "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
                        "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
                        "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos"]:
                state.append(obs.get(key, 0.0))
            state.extend([obs.get("base_x", 0.0), obs.get("base_y", 0.0), obs.get("base_theta", 0.0)])

            # Format for policy - map robot camera names to policy expected names
            # Policy expects all images at 480x640
            img_right = torch.tensor(obs["wrist_right"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_left = torch.tensor(obs["wrist_left"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_top = torch.tensor(obs["top"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_front = torch.tensor(obs["front"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Resize top and front from 720x1280 to 480x640
            img_top = F.interpolate(img_top, size=(480, 640), mode='bilinear', align_corners=False)
            img_front = F.interpolate(img_front, size=(480, 640), mode='bilinear', align_corners=False)

            policy_input = {
                "observation.images.right_wrist": img_right,
                "observation.images.left_wrist": img_left,
                "observation.images.top": img_top,
                "observation.images.front": img_front,
                "observation.state": torch.tensor(state).unsqueeze(0).float(),
            }
            policy_input = {k: v.to(device) for k, v in policy_input.items()}

            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(policy_input)

            # Convert action tensor to dict for robot
            action_array = action.cpu().numpy().flatten()
            action_dict = {
                "left_shoulder_pan.pos": action_array[0],
                "left_shoulder_lift.pos": action_array[1],
                "left_elbow_flex.pos": action_array[2],
                "left_wrist_flex.pos": action_array[3],
                "left_wrist_roll.pos": action_array[4],
                "left_gripper.pos": action_array[5],
                "right_shoulder_pan.pos": action_array[6],
                "right_shoulder_lift.pos": action_array[7],
                "right_elbow_flex.pos": action_array[8],
                "right_wrist_flex.pos": action_array[9],
                "right_wrist_roll.pos": action_array[10],
                "right_gripper.pos": action_array[11],
                "base_vx": action_array[12],
                "base_vy": action_array[13],
                "base_omega": action_array[14],
            }
            robot.send_action(action_dict)

            # 30 FPS timing
            elapsed = time.time() - start
            if elapsed < 1/30:
                time.sleep(1/30 - elapsed)
            else:
                print(f"Warning: Loop took {elapsed*1000:.1f}ms (target: 33.3ms)")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
