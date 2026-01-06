#!/usr/bin/env python

# mimic mathias Desrochers eltopchi1@gmail.com

from dataclasses import dataclass
from typing import Literal

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("mimic_leader")
@dataclass
class MimicLeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str
    
    # Base Control Settings
    # Options: 'keyboard', 'xbox', 'joystick', 'none'
    base_control_mode: str = "keyboard"
    
    # Max speeds for the base
    max_linear_speed: float = 0.1   # m/s
    max_angular_speed: float = 0.2  # rad/s

    #TODO add port for the controller