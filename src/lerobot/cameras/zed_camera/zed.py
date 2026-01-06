
#mimic mathias desrochers eltopchi1@gmail.com

import cv2
import logging
from dataclasses import dataclass
import numpy as np
from typing import Any
from numpy.typing import NDArray
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

logger = logging.getLogger(__name__)

@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(OpenCVCameraConfig):
    """
    Configuration for ZED cameras.
    
    It behaves like a standard camera config, but 'width' and 'height' refer to 
    the SINGLE EYE output (what the robot sees), not the full side-by-side sensor size.
    
    Example:
        ZedCameraConfig(index=4, width=1280, height=720, fps=30)
        # This will internally open the camera at 2560x720, but return 1280x720 images.
    """
    # You can add ZED-specific fields here if needed later (e.g. depth_mode)
    pass


class ZedCamera(OpenCVCamera):
    """
    A subclass of OpenCVCamera specifically for ZED Stereolabs cameras.
    
    It automatically handles the side-by-side splitting so the robot only sees
    the Left Eye (standard view), mimicking a regular webcam.
    """
    def __init__(self, config: ZedCameraConfig):
        super().__init__(config)

    def _configure_capture_settings(self) -> None:
        """
        Overrides the standard setup to handle ZED's specific resolution requirements.
        """
        if not self.is_connected:
            raise ConnectionError(f"Cannot configure settings for {self} as it is not connected.")

        # --- THE ZED HACK ---
        # The user wants a 1280x720 image (Left Eye).
        # But the ZED hardware REJECTS 1280x720. It demands 2560x720 (Side-by-Side).
        # So we force the hardware to open at double the requested width.
        
        req_width = self.config.width
        req_height = self.config.height
        
        # ZED Hardware Expectation: Width must be 2x the single eye width
        hardware_width = req_width * 2
        
        # 1. Set the Hardware Resolution (Side-by-Side)
        self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, hardware_width)
        self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, req_height)
        self.videocapture.set(cv2.CAP_PROP_FPS, self.config.fps)

        # 2. Verify what the hardware actually gave us
        actual_hw_width = int(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_hw_height = int(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 3. "Lie" to the base class
        # The base OpenCVCamera class expects self.capture_width to match the output frame.
        # We set these to the SINGLE EYE dimensions so the validation in _postprocess_image passes.
        self.capture_width = actual_hw_width // 2
        self.capture_height = actual_hw_height
        
        # Update config to match reality (in case hardware fell back to VGA)
        self.width = self.capture_width
        self.height = self.capture_height

        logger.info(f"ZED Camera configured. Hardware: {actual_hw_width}x{actual_hw_height}. Output (Left Eye): {self.width}x{self.height}")

    def read(self) -> NDArray[Any]:
        """
        Reads the frame, chops off the right eye, and returns only the left eye.
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # 1. Grab the raw Side-by-Side frame (e.g., 2560x720)
        ret, frame = self.videocapture.read()

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed.")

        # 2. Crop to Left Eye Only
        # We simply take the left half of the image
        height, width, _ = frame.shape
        left_eye_frame = frame[:, :width // 2]

        # 3. Pass to parent for standard post-processing (Color conversion, Rotation, Validation)
        # The parent will validate this against self.capture_width (which we set to half-width above)
        return self._postprocess_image(left_eye_frame)