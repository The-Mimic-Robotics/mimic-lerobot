#!/usr/bin/env python

# mimic - mathias desrochers eltopchi1@gmail.com

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
# Import your custom base driver
from lerobot.motors.mecanum_base import MecanumBase

from ..robot import Robot
from .config_mimic_follower import MimicFollowerConfig

logger = logging.getLogger(__name__)


class MimicFollower(Robot):
    """
    MimicFollower: Bimanual SO-100 Arms + Custom Mecanum Base.
    Total Action Space: 15 (6 Left + 6 Right + 3 Base)
    """

    config_class = MimicFollowerConfig
    name = "mimic_follower"

    def __init__(self, config: MimicFollowerConfig):
        super().__init__(config)
        self.config = config

        # 1. Configure Left Arm
        left_arm_config = SOFollowerRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )

        # 2. Configure Right Arm
        right_arm_config = SOFollowerRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        # 3. Instantiate Sub-Robots
        self.left_arm = SOFollower(left_arm_config)
        self.right_arm = SOFollower(right_arm_config)
        
        # 4. Instantiate Base
        self.base = MecanumBase(port=config.base_port)
        
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _arm_motors_ft(self) -> dict[str, type]:
        # Enforce Order: Left then Right
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Arms (Position) + Base (Odometry Position x,y,theta) + Cameras
        base_obs_ft = {"base_x": float, "base_y": float, "base_theta": float}
        return {**self._arm_motors_ft, **base_obs_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Arms (Position) + Base (Velocity vx,vy,omega)
        # We use different keys for base action vs observation to be explicit
        base_action_ft = {"base_vx": float, "base_vy": float, "base_omega": float}
        return {**self._arm_motors_ft, **base_action_ft}

    @property
    def is_connected(self) -> bool:
        # Base doesn't typically have an "is_connected" property in simple serial drivers
        # checking if the object exists and serial is open is usually enough
        base_connected = self.base.ser is not None and self.base.ser.is_open
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and base_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        self.base.connect()

        for cam in self.cameras.values():
            cam.connect()

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

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # 1. Left Arm
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # 2. Right Arm
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # 3. Base (Read ODOM)

        #still debating here 

        #TODO clear read
        odom, _ = self.base.read_odom() # Returns [x, y, theta], [vx, vy, omega]
        obs_dict["base_x"] = odom[0]
        obs_dict["base_y"] = odom[1]
        obs_dict["base_theta"] = odom[2]

        # 4. Cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Expects a dictionary containing arm positions AND base velocities.
        """
        # 1. Base Action
        # We use .get() with default 0.0 to be safe
        vx = action.get("base_vx", 0.0)
        vy = action.get("base_vy", 0.0)
        omega = action.get("base_omega", 0.0)
        
        self.base.send_twist(vx, vy, omega)

        # 2. Arm Actions (Splitting)
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back for the return dict
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}
        
        # Return combined actions (useful for logging/verification)
        # We also include the base actions we just sent
        base_feedback = {"base_vx": vx, "base_vy": vy, "base_omega": omega}
        
        return {**prefixed_send_action_left, **prefixed_send_action_right, **base_feedback}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        self.base.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()