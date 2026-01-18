#!/usr/bin/env python3
"""Run trained ACT policy on Mimic robot."""

import time
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.mimic_follower.mimic_follower import MimicFollower


def main():
    # Load policy from HuggingFace
    print("Loading policy from neryotw/act_bimanual_mobile_15d...")
    policy = ACTPolicy.from_pretrained("neryotw/act_bimanual_mobile_15d")
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    print(f"Policy loaded on {device}")

    # Initialize robot
    print("Initializing robot...")
    robot = MimicFollower(
        left_arm_port="/dev/arm_left_follower",
        right_arm_port="/dev/arm_right_follower",
        base_port="/dev/mecanum_base",
        cameras={
            "wrist_right": {"type": "opencv", "index_or_path": "/dev/camera_right_wrist", "width": 640, "height": 480, "fps": 30},
            "wrist_left": {"type": "opencv", "index_or_path": "/dev/camera_left_wrist", "width": 640, "height": 480, "fps": 30},
            "realsense_top": {"type": "zed_camera", "index_or_path": "23081456", "width": 640, "height": 480, "fps": 30},
        }
    )
    robot.connect()
    print("Robot connected!")
    print("\nRunning policy. Press Ctrl+C to stop.\n")

    try:
        while True:
            start = time.time()

            # Get observation
            obs = robot.get_observation()

            # Format for policy
            policy_input = {
                "observation.images.wrist_right": torch.tensor(obs["wrist_right"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                "observation.images.wrist_left": torch.tensor(obs["wrist_left"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                "observation.images.realsense_top": torch.tensor(obs["realsense_top"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                "observation.state": torch.tensor(obs["state"]).unsqueeze(0).float(),
            }
            policy_input = {k: v.to(device) for k, v in policy_input.items()}

            # Get and send action
            with torch.no_grad():
                action = policy.select_action(policy_input)
            robot.send_action(action.cpu().numpy().flatten())

            # 30 FPS
            elapsed = time.time() - start
            if elapsed < 1/30:
                time.sleep(1/30 - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
