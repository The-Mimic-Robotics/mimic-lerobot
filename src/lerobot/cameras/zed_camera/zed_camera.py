
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

# Force OpenCV fallback - ZED SDK causes latency issues
ZED_SDK_AVAILABLE = False
logger.info("ZED camera using OpenCV fallback mode (stereo cropping).")

@CameraConfig.register_subclass("zed_camera")
@dataclass
class ZedCameraConfig(OpenCVCameraConfig):
    """
    Configuration for ZED cameras.
    
    When pyzed is installed: Uses ZED SDK for proper single-eye output at any resolution
    When pyzed is NOT installed: Falls back to OpenCV with manual stereo cropping
    
    'width' and 'height' refer to the SINGLE EYE output (what the robot sees).
    
    Example:
        ZedCameraConfig(index_or_path=0, width=1280, height=720, fps=30)
        # With SDK: Direct 1280x720 left eye
        # Without SDK: 2560x720 stereo → cropped to 1280x720
    """
    use_depth: bool = False  # Future: enable depth map capture
    pass


class ZedCamera(OpenCVCamera):
    """
    ZED Stereolabs camera with automatic SDK/OpenCV fallback.
    
    When pyzed is installed: Uses ZED SDK for proper operation
    When pyzed is NOT installed: Falls back to OpenCV with stereo cropping
    """
    def __init__(self, config: ZedCameraConfig):
        if ZED_SDK_AVAILABLE:
            # Initialize with ZED SDK but still need parent's threading infrastructure
            logger.info("Initializing ZED camera with ZED SDK")
            # Call parent init to set up threading for async_read
            super().__init__(config)
            
            # Override with ZED SDK specific components
            self.zed = sl.Camera()
            self.runtime_params = sl.RuntimeParameters()
            self.image_zed = sl.Mat()
        else:
            # Fall back to OpenCV
            logger.info("Initializing ZED camera with OpenCV fallback")
            super().__init__(config)
    
    @property
    def is_connected(self) -> bool:
        if ZED_SDK_AVAILABLE:
            return self.zed is not None and self.zed.is_opened()
        else:
            return super().is_connected
    
    def connect(self, warmup: bool = True) -> None:
        if ZED_SDK_AVAILABLE:
            # Close any OpenCV capture that might have been opened by parent __init__
            if hasattr(self, 'videocapture') and self.videocapture is not None:
                self.videocapture.release()
                self.videocapture = None
            
            # Use ZED SDK with serial number if provided
            init_params = sl.InitParameters()
            init_params.camera_resolution = self._get_zed_resolution()
            init_params.camera_fps = self.fps
            init_params.depth_mode = sl.DEPTH_MODE.NONE
            
            # If index_or_path is a serial number (int), use it
            if isinstance(self.index_or_path, (int, str)):
                try:
                    serial = int(self.index_or_path) if isinstance(self.index_or_path, str) else self.index_or_path
                    init_params.set_from_serial_number(serial)
                    logger.info(f"Opening ZED camera with serial number: {serial}")
                except (ValueError, TypeError):
                    # If not a valid serial number, try camera ID
                    logger.info("Opening ZED camera with auto-detection")
            else:
                logger.info("Opening ZED camera with auto-detection")
            
            err = self.zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise ConnectionError(f"Failed to open ZED camera: {err}")
            
            cam_info = self.zed.get_camera_information()
            actual_res = cam_info.camera_configuration.resolution
            logger.info(f"ZED Camera connected: {actual_res.width}x{actual_res.height} @ {self.fps}fps (S/N: {cam_info.serial_number})")
            
            # Start the read thread for async_read support
            self._start_read_thread()
            
            if warmup and self.warmup_s > 0:
                import time
                start_time = time.time()
                while time.time() - start_time < self.warmup_s:
                    self.read()
                    time.sleep(0.1)
        else:
            # Use OpenCV fallback
            super().connect(warmup)
    
    def _get_zed_resolution(self):
        """Map requested resolution to ZED SDK resolution enum."""
        # ZED SDK supports specific resolutions
        res_map = {
            (2208, 1242): sl.RESOLUTION.HD2K,
            (1920, 1080): sl.RESOLUTION.HD1080,
            (1280, 720): sl.RESOLUTION.HD720,
            (672, 376): sl.RESOLUTION.VGA,
        }
        key = (self.width, self.height)
        if key in res_map:
            return res_map[key]
        # Default to closest match
        logger.warning(f"Resolution {self.width}x{self.height} not standard, using HD720")
        return sl.RESOLUTION.HD720
    
    def read(self, color_mode=None) -> NDArray[Any]:
        if ZED_SDK_AVAILABLE:
            # Use ZED SDK
            if not self.is_connected:
                raise ConnectionError(f"{self} is not connected.")
            
            err = self.zed.grab(self.runtime_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED grab failed: {err}")
            
            # Retrieve left eye image
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            frame = self.image_zed.get_data()
            
            # Convert BGRA to RGB/BGR based on color_mode
            requested_color_mode = self.color_mode if color_mode is None else color_mode
            if frame.shape[2] == 4:  # BGRA
                if requested_color_mode == self.color_mode.RGB:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            return frame[:, :, :3]  # Remove alpha channel if present
        else:
            # Use OpenCV fallback with stereo cropping
            if not self.is_connected:
                raise ConnectionError(f"{self} is not connected.")

            ret, frame = self.videocapture.read()
            if not ret or frame is None:
                raise RuntimeError(f"{self} read failed.")

            # Crop to Left Eye Only
            height, width, _ = frame.shape
            left_eye_frame = frame[:, :width // 2]

            return self._postprocess_image(left_eye_frame, color_mode)
    
    def disconnect(self) -> None:
        if ZED_SDK_AVAILABLE:
            # Stop the read thread first to prevent segfault
            self._stop_read_thread()
            
            if self.zed is not None:
                self.zed.close()
            logger.info(f"{self} disconnected.")
        else:
            super().disconnect()
    
    # Only override these if using OpenCV fallback
    def _configure_capture_settings(self) -> None:
        if not ZED_SDK_AVAILABLE:
            # Only needed for OpenCV fallback
            if not self.is_connected:
                raise ConnectionError(f"Cannot configure settings for {self} as it is not connected.")

            req_width = self.config.width
            req_height = self.config.height
            hardware_width = req_width * 2
            
            self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, hardware_width)
            self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, req_height)
            self.videocapture.set(cv2.CAP_PROP_FPS, self.config.fps)

            actual_hw_width = int(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_hw_height = int(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.capture_width = actual_hw_width // 2
            self.capture_height = actual_hw_height
            self.width = self.capture_width
            self.height = self.capture_height

            logger.info(f"ZED Camera configured (OpenCV). Hardware: {actual_hw_width}x{actual_hw_height}. Output (Left Eye): {self.width}x{self.height}")
