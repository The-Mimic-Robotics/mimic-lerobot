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

# Try importing pygame for gamepad/joystick
PYGAME_AVAILABLE = True
try:
    import pygame
    import pygame.joystick
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
    logger.info("pygame library not available. Install with: pip install pygame")


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
    """Simple Xbox controller using pygame - loads config from YAML."""
    
    def __init__(self, config_path: str = None, **kwargs):
        # Load config from YAML if provided
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                teleop = config['teleop_node']['ros__parameters']
                
                # Extract settings from YAML
                max_linear_speed = teleop['scale_linear']['x']
                max_angular_speed = teleop['scale_angular']['yaw']
                max_linear_speed_turbo = teleop['scale_linear_turbo']['x']
                max_angular_speed_turbo = teleop['scale_angular_turbo']['yaw']
                self.enable_button = teleop['enable_button']
                self.enable_turbo_button = teleop['enable_turbo_button']
                self.deadzone = config['joy_node']['ros__parameters']['deadzone']
                
                # Axis mappings
                self.axis_linear_x = teleop['axis_linear']['x']
                self.axis_linear_y = teleop['axis_linear']['y']
                self.axis_angular_yaw = teleop['axis_angular']['yaw']
        else:
            # Use kwargs or defaults
            max_linear_speed = kwargs.get('max_linear_speed', 0.1)
            max_angular_speed = kwargs.get('max_angular_speed', 0.2)
            max_linear_speed_turbo = kwargs.get('max_linear_speed_turbo', 0.2)
            max_angular_speed_turbo = kwargs.get('max_angular_speed_turbo', 0.4)
            self.enable_button = kwargs.get('enable_button', 4)  # LB = safety
            self.enable_turbo_button = kwargs.get('enable_turbo_button', 5)  # RB = turbo
            self.deadzone = kwargs.get('deadzone', 0.15)
            self.axis_linear_x = 1  # Left stick Y (forward/back)
            self.axis_linear_y = 0  # Left stick X (strafe left/right)
            self.axis_angular_yaw = 3  # Right stick X (rotation)
        
        super().__init__(max_linear_speed, max_angular_speed)
        self.max_linear_speed_turbo = max_linear_speed_turbo
        self.max_angular_speed_turbo = max_angular_speed_turbo
        self.joystick = None
        
        if not PYGAME_AVAILABLE:
            logger.error("pygame not available. Install: conda install pygame -c conda-forge")
    
    def connect(self) -> None:
        if not PYGAME_AVAILABLE:
            return
        
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            logger.error("No gamepad found")
            return
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logger.info(f"Connected: {self.joystick.get_name()}")
    
    def disconnect(self) -> None:
        if self.joystick:
            self.joystick.quit()
        pygame.joystick.quit()
    
    def get_velocities(self) -> Tuple[float, float, float]:
        if not self.joystick:
            return 0.0, 0.0, 0.0
        
        pygame.event.pump()
        
        # Read buttons: LB=4 (enable normal speed), RB=5 (enable turbo speed)
        enable_normal = self.joystick.get_button(self.enable_button)
        enable_turbo = self.joystick.get_button(self.enable_turbo_button)
        
        # SAFETY: Must hold at least one enable button (either normal OR turbo)
        if not enable_normal and not enable_turbo:
            return 0.0, 0.0, 0.0
        
        # Determine which speed to use: turbo if RB pressed, normal if only LB pressed
        use_turbo = enable_turbo
        
        # Read axes
        left_x = self.joystick.get_axis(self.axis_linear_y)
        left_y = self.joystick.get_axis(self.axis_linear_x)
        right_x = self.joystick.get_axis(self.axis_angular_yaw)
        
        # Apply deadzone
        if abs(left_x) < self.deadzone: left_x = 0.0
        if abs(left_y) < self.deadzone: left_y = 0.0
        if abs(right_x) < self.deadzone: right_x = 0.0
        
        # Select speed: turbo if RB pressed, normal otherwise
        linear_speed = self.max_linear_speed_turbo if use_turbo else self.max_linear_speed
        angular_speed = self.max_angular_speed_turbo if use_turbo else self.max_angular_speed
        
        # Calculate velocities
        vx = -left_y * linear_speed  # Forward/back (axis 1)
        vy = -left_x * linear_speed  # Strafe left/right (axis 0) - NEGATED to fix reversed direction
        omega = -right_x * angular_speed  # Rotation (axis 3)
        
        return vx, vy, omega
    
    def is_connected(self) -> bool:
        return self.joystick is not None


class JoystickBaseController(BaseController):
    """Generic joystick-based base controller using pygame library."""
    
    def __init__(self, max_linear_speed: float = 0.1, max_angular_speed: float = 0.2,
                 max_linear_speed_turbo: float = 0.2, max_angular_speed_turbo: float = 0.4,
                 device_path: str = None, enable_button: int = 6, enable_turbo_button: int = 7,
                 use_toggle_turbo: bool = True, require_safety_button: bool = False):
        super().__init__(max_linear_speed, max_angular_speed)
        self.device_index = 0
        self.joystick = None
        self.deadzone = 0.15
        
        # Speed settings
        self.max_linear_speed_turbo = max_linear_speed_turbo
        self.max_angular_speed_turbo = max_angular_speed_turbo
        
        # Button states
        self.enable_button = enable_button
        self.enable_turbo_button = enable_turbo_button
        
        # Toggle turbo mode feature
        self.use_toggle_turbo = use_toggle_turbo  # Enable/disable toggle functionality
        self.require_safety_button = require_safety_button  # Require safety button to be held
        self.turbo_toggled = False  # Current toggle state (turbo on/off)
        self.both_buttons_pressed_last = False  # Track previous state for edge detection
        
        if not PYGAME_AVAILABLE:
            logger.warning("pygame library not available. Joystick controller will not work.")
    
    def connect(self) -> None:
        if not PYGAME_AVAILABLE:
            logger.error("Cannot connect joystick: pygame library not available")
            logger.info("Install with: pip install pygame")
            return
        
        try:
            if not pygame.get_init():
                pygame.init()
            pygame.joystick.init()
            
            joystick_count = pygame.joystick.get_count()
            if joystick_count == 0:
                logger.error("No joystick devices found")
                return
            
            self.joystick = pygame.joystick.Joystick(self.device_index)
            self.joystick.init()
            
            logger.info(f"Joystick connected: {self.joystick.get_name()}")
            logger.info(f"  Axes: {self.joystick.get_numaxes()}, Buttons: {self.joystick.get_numbuttons()}")
        
        except Exception as e:
            logger.error(f"Failed to connect joystick: {e}")
            self.joystick = None
    
    def disconnect(self) -> None:
        if self.joystick:
            self.joystick.quit()
            self.joystick = None
        pygame.joystick.quit()
        logger.info("Joystick disconnected")
    
    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value
    
    def get_velocities(self) -> Tuple[float, float, float]:
        if not self.joystick or not PYGAME_AVAILABLE:
            return 0.0, 0.0, 0.0
        
        try:
            pygame.event.pump()
            
            # Read button states
            enable_button_pressed = self.joystick.get_button(self.enable_button)
            turbo_button_pressed = self.joystick.get_button(self.enable_turbo_button)
            
            # SAFETY: Only output velocities if enable button is pressed (if required)
            if self.require_safety_button and not enable_button_pressed:
                return 0.0, 0.0, 0.0
            
            # Handle turbo mode selection
            use_turbo = False
            
            if self.use_toggle_turbo:
                # Toggle mode: Both buttons pressed together toggles turbo on/off
                both_buttons_pressed = enable_button_pressed and turbo_button_pressed
                
                # Detect rising edge (buttons just pressed together)
                if both_buttons_pressed and not self.both_buttons_pressed_last:
                    self.turbo_toggled = not self.turbo_toggled
                    logger.info(f"Turbo mode toggled: {'ON' if self.turbo_toggled else 'OFF'}")
                
                self.both_buttons_pressed_last = both_buttons_pressed
                use_turbo = self.turbo_toggled
            else:
                # Original behavior: Turbo only when turbo button is held
                use_turbo = turbo_button_pressed
            
            # Read joystick axes
            axis_x = self.joystick.get_axis(0)
            axis_y = self.joystick.get_axis(1)
            axis_twist = self.joystick.get_axis(2) if self.joystick.get_numaxes() > 2 else 0.0
            
            # Apply deadzones
            axis_x = self._apply_deadzone(axis_x)
            axis_y = self._apply_deadzone(axis_y)
            axis_twist = self._apply_deadzone(axis_twist)
            
            # Select speed based on turbo mode
            if use_turbo:
                linear_speed = self.max_linear_speed_turbo
                angular_speed = self.max_angular_speed_turbo
            else:
                linear_speed = self.max_linear_speed
                angular_speed = self.max_angular_speed
            
            # Map to robot velocities
            vx = -axis_y * linear_speed
            vy = axis_x * linear_speed
            omega = -axis_twist * angular_speed
            
            return vx, vy, omega
        
        except Exception as e:
            logger.debug(f"Error reading joystick: {e}")
            return 0.0, 0.0, 0.0
    
    def is_connected(self) -> bool:
        return PYGAME_AVAILABLE and self.joystick is not None and self.joystick.get_init()


def create_base_controller(
    mode: str,
    max_linear_speed: float = 0.1,
    max_angular_speed: float = 0.2,
    max_linear_speed_turbo: float = 0.2,
    max_angular_speed_turbo: float = 0.4,
    device_path: str = None,
    enable_button: int = None,
    enable_turbo_button: int = None,
    config_path: str = None
) -> BaseController:
    """
    Factory function to create appropriate base controller.
    
    Args:
        mode: "keyboard", "xbox", or "joystick"
        max_linear_speed: Maximum linear velocity (m/s)
        max_angular_speed: Maximum angular velocity (rad/s)
        max_linear_speed_turbo: Maximum linear velocity with turbo (m/s)
        max_angular_speed_turbo: Maximum angular velocity with turbo (rad/s)
        device_path: Device path for gamepad/joystick (e.g., "/dev/input/js0")
        enable_button: Button number for safety enable (LB=4 for Xbox, 6 for joystick)
        enable_turbo_button: Button number for turbo mode (RB=5 for Xbox, 7 for joystick)
        config_path: Path to YAML config file (for Xbox controller)
    
    Returns:
        BaseController instance
    """
    if mode == "keyboard":
        return KeyboardBaseController(max_linear_speed, max_angular_speed)
    
    elif mode == "xbox":
        # If config_path provided, load from YAML. Otherwise use kwargs
        if config_path:
            return XboxBaseController(config_path=config_path)
        else:
            return XboxBaseController(
                max_linear_speed=max_linear_speed,
                max_angular_speed=max_angular_speed,
                max_linear_speed_turbo=max_linear_speed_turbo,
                max_angular_speed_turbo=max_angular_speed_turbo,
                enable_button=enable_button if enable_button is not None else 4,  # LB
                enable_turbo_button=enable_turbo_button if enable_turbo_button is not None else 5  # RB
            )
    
    elif mode == "joystick":
        return JoystickBaseController(
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            max_linear_speed_turbo=max_linear_speed_turbo,
            max_angular_speed_turbo=max_angular_speed_turbo,
            enable_button=enable_button if enable_button is not None else 6,
            enable_turbo_button=enable_turbo_button if enable_turbo_button is not None else 7,
            use_toggle_turbo=True,  # Toggle turbo by pressing both buttons simultaneously
            require_safety_button=False  # No safety button required for joystick
        )
    
    else:
        raise ValueError(f"Unknown base control mode: {mode}. Use 'keyboard', 'xbox', or 'joystick'")
