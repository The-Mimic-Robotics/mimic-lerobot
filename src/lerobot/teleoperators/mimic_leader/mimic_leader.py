#!/usr/bin/env python

# mimic mathias Desrochers eltopchi1@gmail.com

import logging
from functools import cached_property

# Import standard Leader parts
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader

from ..teleoperator import Teleoperator
from .config_mimic_leader import MimicLeaderConfig
from .base_controllers import create_base_controller

logger = logging.getLogger(__name__)


class MimicLeader(Teleoperator):
    """
    MimicLeader: Bimanual SO-100 Leaders + Base Control (Keyboard/Xbox).
    """

    config_class = MimicLeaderConfig
    name = "mimic_leader"

    def __init__(self, config: MimicLeaderConfig):
        super().__init__(config)
        self.config = config

        # 1. Setup Left Arm
        left_arm_config = SO100LeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
        )

        # 2. Setup Right Arm
        right_arm_config = SO100LeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
        )

        self.left_arm = SO100Leader(left_arm_config)
        self.right_arm = SO100Leader(right_arm_config)

        # 3. Setup Base Controller based on mode
        self.base_controller = create_base_controller(
            mode=config.base_control_mode,
            max_linear_speed=config.max_linear_speed,
            max_angular_speed=config.max_angular_speed,
            device_path="/dev/input/js0"  # Default gamepad/joystick path
        )
        logger.info(f"Base control mode: {config.base_control_mode}")

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Arm Features
        arm_feats = {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }
        # Base Features (Velocity Commands)
        base_feats = {"base_vx": float, "base_vy": float, "base_omega": float}
        return {**arm_feats, **base_feats}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        arms_connected = self.left_arm.is_connected and self.right_arm.is_connected
        base_connected = self.base_controller.is_connected()
        return arms_connected and base_connected

    def connect(self, calibrate: bool = True) -> None:
        # Connect Arms
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        
        # Connect base controller
        self.base_controller.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        """Get combined action from both arms and base controller."""
        action_dict = {}

        # Add "left_" prefix for left arm actions
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix for right arm actions
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        # Add base velocities
        vx, vy, omega = self.base_controller.get_velocities()
        action_dict.update({
            "base_vx": vx,
            "base_vy": vy,
            "base_omega": omega,
        })

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Remove "left_" prefix
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        self.base_controller.disconnect()