#!/usr/bin/env python3
"""Run trained ACT policy on Mimic robot."""

import time
import cv2
import numpy as np
import torch
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.robots.mimic_follower.mimic_follower import MimicFollower


# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def normalize_state(state: np.ndarray, stats: dict, eps: float = 1e-8) -> torch.Tensor:
    """Normalize state using mean/std with safeguards for zero std."""
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    # Use minimum std of 1.0 to avoid division by near-zero
    std = torch.clamp(std, min=1.0)
    state_tensor = torch.tensor(state, dtype=torch.float32)
    return (state_tensor - mean) / std


def unnormalize_action(action: torch.Tensor, stats: dict) -> torch.Tensor:
    """Unnormalize action back to original scale."""
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=action.device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=action.device)
    # Use minimum std of 1.0 to match normalization
    std = torch.clamp(std, min=1.0)
    return action * std + mean


def normalize_image(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize image using ImageNet stats."""
    # Convert to float [0, 1]
    img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
    # Permute to (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    # Normalize with ImageNet stats
    mean = IMAGENET_MEAN.view(3, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    # Add batch dimension and move to device
    return img_tensor.unsqueeze(0).to(device)


def main():
    # Load policy from HuggingFace
    print("Loading policy from neryotw/act_bimanual_drift...")
    policy = ACTPolicy.from_pretrained("neryotw/act_bimanual_drift")
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    print(f"Policy loaded on {device}")
    print(f"Policy config: chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}")

    # Enable temporal ensembling for smooth, stable actions
    # Coefficient 0.01 = more weight on older predictions (smoother, more stable)
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    policy.config.temporal_ensemble_coeff = 0.01
    policy.config.n_action_steps = 1  # Required for temporal ensembling
    policy.temporal_ensembler = ACTTemporalEnsembler(
        temporal_ensemble_coeff=0.01,
        chunk_size=policy.config.chunk_size,
    )
    policy._action_queue.clear()
    print(f"Enabled temporal ensembling (coeff=0.01) for smoother actions")

    # Load dataset metadata for normalization stats
    print("Loading normalization stats from dataset...")
    dataset_metadata = LeRobotDatasetMetadata("neryotw/bimanual_blue_block_handover_1_drift")
    state_stats = dataset_metadata.stats["observation.state"]
    action_stats = dataset_metadata.stats["action"]
    print("Normalization stats loaded.")

    # Get state feature names in order
    state_names = dataset_metadata.features["observation.state"]["names"]
    action_names = dataset_metadata.features["action"]["names"]
    print(f"State features: {state_names}")
    print(f"Action features: {action_names}")

    # Initialize robot (without front camera - it has broken stats)
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

    print("\nRunning policy at 30 FPS. Press Ctrl+C to stop.\n")
    debug_first = True
    step_count = 0

    try:
        while True:
            step_count += 1
            start = time.time()

            # Get observation
            obs = robot.get_observation()

            # Build state vector in correct order
            state = np.array([obs[name] for name in state_names], dtype=np.float32)

            # Normalize state
            state_normalized = normalize_state(state, state_stats)

            # Process images
            img_right = normalize_image(obs["wrist_right"], device)
            img_left = normalize_image(obs["wrist_left"], device)
            # Resize top/front cameras from 720x1280 to 480x640
            img_top = normalize_image(cv2.resize(obs["top"], (640, 480)), device)
            img_front = normalize_image(cv2.resize(obs["front"], (640, 480)), device)

            # Build observation dict for policy
            policy_input = {
                "observation.state": state_normalized.unsqueeze(0).to(device),
                "observation.images.right_wrist": img_right,
                "observation.images.left_wrist": img_left,
                "observation.images.top": img_top,
                "observation.images.front": img_front,
            }

            # Debug first iteration
            if debug_first:
                print("=== DEBUG: First iteration ===")
                print(f"Raw state (first 5): {state[:5]}")
                print(f"Normalized state (first 5): {state_normalized[:5].numpy()}")
                print(f"State stats mean (first 5): {state_stats['mean'][:5]}")
                print(f"State stats std (first 5): {state_stats['std'][:5]}")
                for k, v in policy_input.items():
                    print(f"  {k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}")

            # Get action from policy (temporal ensembling predicts every step)
            with torch.no_grad():
                action = policy.select_action(policy_input)

            if debug_first:
                print(f"\nRaw policy output: shape={action.shape}, values={action.cpu().numpy().flatten()[:5]}...")

            # Unnormalize action
            action = unnormalize_action(action.squeeze(0), action_stats)

            if debug_first:
                print(f"Unnormalized action (first 5): {action.cpu().numpy()[:5]}")

            # Build action dict for robot
            action_array = action.cpu().numpy()
            action_dict = {}
            for i, name in enumerate(action_names):
                action_dict[name] = float(action_array[i])

            if debug_first:
                print("\nAction dict sent to robot:")
                for k, v in action_dict.items():
                    print(f"  {k}: {v:.2f}")
                debug_first = False
                print("=== END DEBUG ===\n")

            robot.send_action(action_dict)

            # Show gripper state periodically
            if step_count % 30 == 0:  # Every ~1 second
                print(f"Step {step_count}: L_grip={action_dict['left_gripper.pos']:.1f}, R_grip={action_dict['right_gripper.pos']:.1f}, L_elbow={action_dict['left_elbow_flex.pos']:.1f}, R_elbow={action_dict['right_elbow_flex.pos']:.1f}")

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
