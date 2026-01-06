#!/usr/bin/env python

# mimic mathias Desrochers eltopchi1@gmail.com

#TODO add xbox ctr, and joystick implementation

import logging
import time
from functools import cached_property
import os
import sys

# Import standard Leader parts
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader

from ..teleoperator import Teleoperator
from .config_mimic_leader import MimicLeaderConfig

logger = logging.getLogger(__name__)

# Handle Pynput import safely (like the original keyboard teleop)
PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")
    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


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

        # 3. Setup Base Control State
        self.base_pressed_keys = set()
        self.key_listener = None
        
        if self.config.base_control_mode == "keyboard" and not PYNPUT_AVAILABLE:
            logger.warning("Keyboard control requested but pynput not available. Base will not move.")

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
        if self.config.base_control_mode == "keyboard":
            return arms_connected and self.key_listener is not None and self.key_listener.is_alive()
        return arms_connected

    def connect(self, calibrate: bool = True) -> None:
        # Connect Arms
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        
        # Connect Keyboard Listener if needed
        if self.config.base_control_mode == "keyboard" and PYNPUT_AVAILABLE:
            if self.key_listener is None:
                self.key_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release
                )
                self.key_listener.start()
                logger.info("MimicLeader: Keyboard listener started for Base control.")

    def _on_key_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.base_pressed_keys.add(key.char.lower())
        except AttributeError:
            pass

    def _on_key_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.base_pressed_keys.discard(key.char.lower())
        except AttributeError:
            pass

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

    def _get_base_action(self) -> dict[str, float]:
        vx, vy, omega = 0.0, 0.0, 0.0
        
        if self.config.base_control_mode == "keyboard":
            # Simple Mecanum Mapping
            # W/S = Forward/Back (X)
            if 'w' in self.base_pressed_keys: vx += self.config.max_linear_speed
            if 's' in self.base_pressed_keys: vx -= self.config.max_linear_speed
            
            # A/D = Left/Right (Y - Strafing)
            if 'a' in self.base_pressed_keys: vy += self.config.max_linear_speed
            if 'd' in self.base_pressed_keys: vy -= self.config.max_linear_speed
            
            # Q/E = Rotate (Omega)
            if 'q' in self.base_pressed_keys: omega += self.config.max_angular_speed
            if 'e' in self.base_pressed_keys: omega -= self.config.max_angular_speed
            
        elif self.config.base_control_mode == "xbox":
            # Placeholder for XBox Controller
            # TODO: Implement XInput or pygame joystick logic here
            pass
            
        elif self.config.base_control_mode == "joystick":
            # Placeholder for generic joystick
            pass

        return {"base_vx": vx, "base_vy": vy, "base_omega": omega}

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        # 1. Get Base Action
        base_action = self._get_base_action()
        action_dict.update(base_action)

        # 2. Get Arm Actions
        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

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
        
        if self.key_listener:
            self.key_listener.stop()
            self.key_listener = None