#!/usr/bin/env python3
"""Run trained ACT policy on Mimic robot."""

import time
import torch
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.robots.mimic_follower.mimic_follower import MimicFollower


def main():
    # Load policy from HuggingFace
    print("Loading policy from neryotw/act_bimanual_drift...")
    policy = ACTPolicy.from_pretrained("neryotw/act_bimanual_drift")
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    print(f"Policy loaded on {device}")

    # Load dataset metadata for normalization stats
    print("Loading normalization stats from dataset...")
    dataset_metadata = LeRobotDatasetMetadata("neryotw/bimanual_blue_block_handover_1_drift")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_metadata.stats
    )
    print("Normalization stats loaded.")

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

            # Remap camera keys from robot format to policy format
            obs_remapped = {}
            for k, v in obs.items():
                if k == "wrist_right":
                    obs_remapped["right_wrist"] = v
                elif k == "wrist_left":
                    obs_remapped["left_wrist"] = v
                else:
                    obs_remapped[k] = v

            # Build inference frame using LeRobot helper
            obs_frame = build_inference_frame(
                observation=obs_remapped,
                ds_features=dataset_metadata.features,
                device=device
            )

            # Preprocess (normalizes state and images)
            obs_normalized = preprocess(obs_frame)

            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(obs_normalized)

            # Postprocess (unnormalizes action back to degrees)
            action = postprocess(action)

            # Convert action to robot format
            action_dict = make_robot_action(action, dataset_metadata.features)
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
