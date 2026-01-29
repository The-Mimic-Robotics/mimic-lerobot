#!/usr/bin/env python3
"""Run trained ACT policy on Mimic robot."""

import time
import torch
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
        }
    )
    robot = MimicFollower(config)
    robot.connect(calibrate=False)
    print("Robot connected!")
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
            policy_input = {
                "observation.images.right_wrist": torch.tensor(obs["wrist_right"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                "observation.images.left_wrist": torch.tensor(obs["wrist_left"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                "observation.images.top": torch.tensor(obs["top"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
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
