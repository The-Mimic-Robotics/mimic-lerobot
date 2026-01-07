#!/usr/bin/env python
"""
Base control handlers for MimicLeader teleoperator.
Supports keyboard, Xbox controller, and custom joystick inputs.
"""

import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Tuple

logger = logging.getLogger(__name__)

# Try importing pynput for keyboard
PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        raise ImportError("pynput blocked intentionally due to no display.")
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False

# Try importing inputs library for gamepad/joystick
INPUTS_AVAILABLE = True
try:
    import inputs
except ImportError:
    inputs = None
    INPUTS_AVAILABLE = False
    logger.info("inputs library not available. Install with: pip install inputs")


class BaseController(ABC):
    """Abstract base class for base movement controllers."""
    
    def __init__(self, max_linear_speed: float = 0.1, max_angular_speed: float = 0.2):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
    
    @abstractmethod
    def connect(self) -> None:
        """Initialize and connect the controller."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Clean up and disconnect the controller."""
        pass
    
    @abstractmethod
    def get_velocities(self) -> Tuple[float, float, float]:
        """
        Get base velocities from controller.
        Returns: (vx, vy, omega) tuple
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if controller is connected and ready."""
        pass


class KeyboardBaseController(BaseController):
    """Keyboard-based base controller using pynput."""
    
    def __init__(self, max_linear_speed: float = 0.1, max_angular_speed: float = 0.2):
        super().__init__(max_linear_speed, max_angular_speed)
        self.pressed_keys = set()
        self.key_listener = None
        
        if not PYNPUT_AVAILABLE:
            logger.warning("pynput not available. Keyboard control will not work.")
    
    def connect(self) -> None:
        if not PYNPUT_AVAILABLE:
            logger.error("Cannot connect keyboard controller: pynput not available")
            return
        
        if self.key_listener is None:
            self.key_listener = pynput_keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.key_listener.start()
            logger.info("Keyboard controller connected")
    
    def disconnect(self) -> None:
        if self.key_listener is not None:
            self.key_listener.stop()
            self.key_listener = None
            logger.info("Keyboard controller disconnected")
    
    def _on_key_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.add(key.char.lower())
        except AttributeError:
            pass
    
    def _on_key_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            pass
    
    def get_velocities(self) -> Tuple[float, float, float]:
        vx, vy, omega = 0.0, 0.0, 0.0
        
        # W/S = Forward/Back (X)
        if 'w' in self.pressed_keys:
            vx += self.max_linear_speed
        if 's' in self.pressed_keys:
            vx -= self.max_linear_speed
        
        # A/D = Left/Right (Y - Strafing)
        if 'a' in self.pressed_keys:
            vy += self.max_linear_speed
        if 'd' in self.pressed_keys:
            vy -= self.max_linear_speed
        
        # Q/E = Rotate (Omega)
        if 'q' in self.pressed_keys:
            omega += self.max_angular_speed
        if 'e' in self.pressed_keys:
            omega -= self.max_angular_speed
        
        return vx, vy, omega
    
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and self.key_listener is not None and self.key_listener.is_alive()


class XboxBaseController(BaseController):
    """Xbox controller-based base controller using inputs library."""
    
    def __init__(self, max_linear_speed: float = 0.1, max_angular_speed: float = 0.2, device_path: str = None):
        super().__init__(max_linear_speed, max_angular_speed)
        self.device_path = device_path  # e.g., "/dev/input/js0"
        self.gamepad = None
        self.left_stick_x = 0.0
        self.left_stick_y = 0.0
        self.right_stick_x = 0.0
        self.deadzone = 0.15  # Ignore inputs below this threshold
        
        if not INPUTS_AVAILABLE:
            logger.warning("inputs library not available. Xbox controller will not work.")
    
    def connect(self) -> None:
        if not INPUTS_AVAILABLE:
            logger.error("Cannot connect Xbox controller: inputs library not available")
            logger.info("Install with: pip install inputs")
            return
        
        try:
            # Try to find Xbox controller
            devices = inputs.devices.gamepads
            if not devices:
                logger.error("No gamepad devices found")
                return
            
            # Use first gamepad found
            # Note: inputs library returns device name (e.g., "Microsoft Xbox Series S|X Controller")
            # when converting to string, not the /dev/input/js0 path
            self.gamepad = devices[0]
            logger.info(f"Xbox controller connected: {self.gamepad}")
        
        except Exception as e:
            logger.error(f"Failed to connect Xbox controller: {e}")
            self.gamepad = None
    
    def disconnect(self) -> None:
        self.gamepad = None
        logger.info("Xbox controller disconnected")
    
    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value
    
    def get_velocities(self) -> Tuple[float, float, float]:
        if not self.gamepad or not INPUTS_AVAILABLE:
            return 0.0, 0.0, 0.0
        
        try:
            # Read all pending events
            events = inputs.get_gamepad()
            for event in events:
                if event.ev_type == "Absolute":
                    # Xbox controller mapping
                    if event.code == "ABS_X":  # Left stick X
                        self.left_stick_x = event.state / 32768.0
                    elif event.code == "ABS_Y":  # Left stick Y
                        self.left_stick_y = event.state / 32768.0
                    elif event.code == "ABS_RX":  # Right stick X
                        self.right_stick_x = event.state / 32768.0
        
        except Exception as e:
            logger.debug(f"Error reading Xbox controller: {e}")
        
        # Apply deadzones
        left_x = self._apply_deadzone(self.left_stick_x)
        left_y = self._apply_deadzone(self.left_stick_y)
        right_x = self._apply_deadzone(self.right_stick_x)
        
        # Map to robot velocities
        # Left stick: forward/back and strafe
        # Right stick X: rotation
        vx = -left_y * self.max_linear_speed  # Invert Y for forward
        vy = -left_x * self.max_linear_speed  # Invert X for strafe right
        omega = -right_x * self.max_angular_speed  # Right stick for rotation
        
        return vx, vy, omega
    
    def is_connected(self) -> bool:
        return INPUTS_AVAILABLE and self.gamepad is not None


class JoystickBaseController(BaseController):
    """Custom joystick-based base controller using inputs library."""
    
    def __init__(self, max_linear_speed: float = 0.1, max_angular_speed: float = 0.2, device_path: str = None):
        super().__init__(max_linear_speed, max_angular_speed)
        self.device_path = device_path
        self.joystick = None
        self.axis_x = 0.0
        self.axis_y = 0.0
        self.axis_twist = 0.0
        self.deadzone = 0.15
        
        if not INPUTS_AVAILABLE:
            logger.warning("inputs library not available. Joystick control will not work.")
    
    def connect(self) -> None:
        if not INPUTS_AVAILABLE:
            logger.error("Cannot connect joystick: inputs library not available")
            logger.info("Install with: pip install inputs")
            return
        
        try:
            # Try to find joystick
            devices = inputs.devices.gamepads
            if not devices:
                logger.error("No joystick devices found")
                return
            
            # Use first device, or specified device
            if self.device_path:
                for device in devices:
                    if self.device_path in str(device):
                        self.joystick = device
                        break
            else:
                self.joystick = devices[0]
            
            if self.joystick:
                logger.info(f"Joystick connected: {self.joystick}")
            else:
                logger.error(f"Joystick not found at {self.device_path}")
        
        except Exception as e:
            logger.error(f"Failed to connect joystick: {e}")
            self.joystick = None
    
    def disconnect(self) -> None:
        self.joystick = None
        logger.info("Joystick disconnected")
    
    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value
    
    def get_velocities(self) -> Tuple[float, float, float]:
        if not self.joystick or not INPUTS_AVAILABLE:
            return 0.0, 0.0, 0.0
        
        try:
            # Read all pending events
            events = inputs.get_gamepad()
            for event in events:
                if event.ev_type == "Absolute":
                    # Generic joystick mapping (may need adjustment)
                    if event.code == "ABS_X":
                        self.axis_x = event.state / 32768.0
                    elif event.code == "ABS_Y":
                        self.axis_y = event.state / 32768.0
                    elif event.code == "ABS_RZ" or event.code == "ABS_Z":  # Twist axis
                        self.axis_twist = event.state / 32768.0
        
        except Exception as e:
            logger.debug(f"Error reading joystick: {e}")
        
        # Apply deadzones
        x = self._apply_deadzone(self.axis_x)
        y = self._apply_deadzone(self.axis_y)
        twist = self._apply_deadzone(self.axis_twist)
        
        # Map to robot velocities
        vx = -y * self.max_linear_speed
        vy = -x * self.max_linear_speed
        omega = -twist * self.max_angular_speed
        
        return vx, vy, omega
    
    def is_connected(self) -> bool:
        return INPUTS_AVAILABLE and self.joystick is not None


def create_base_controller(
    mode: str,
    max_linear_speed: float = 0.1,
    max_angular_speed: float = 0.2,
    device_path: str = None
) -> BaseController:
    """
    Factory function to create appropriate base controller.
    
    Args:
        mode: "keyboard", "xbox", or "joystick"
        max_linear_speed: Maximum linear velocity (m/s)
        max_angular_speed: Maximum angular velocity (rad/s)
        device_path: Device path for gamepad/joystick (e.g., "/dev/input/js0")
    
    Returns:
        BaseController instance
    """
    if mode == "keyboard":
        return KeyboardBaseController(max_linear_speed, max_angular_speed)
    elif mode == "xbox":
        return XboxBaseController(max_linear_speed, max_angular_speed, device_path)
    elif mode == "joystick":
        return JoystickBaseController(max_linear_speed, max_angular_speed, device_path)
    else:
        raise ValueError(f"Unknown base control mode: {mode}. Use 'keyboard', 'xbox', or 'joystick'")
