#!/usr/bin/env python

# mimic - mathias desrochers eltopchi1@gmail.com

from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("mimic_follower")
@dataclass
class MimicFollowerConfig(RobotConfig):
    left_arm_port: str
    right_arm_port: str
    base_port: str  # Added for the Mecanum Base

    # Optional
    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: float | dict[str, float] | None = None
    left_arm_use_degrees: bool = False
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_use_degrees: bool = False

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)