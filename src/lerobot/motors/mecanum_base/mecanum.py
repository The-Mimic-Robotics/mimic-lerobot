#mimic team, mathias Desrochers eltopchi1@gmail.com

import serial
import time
import logging
import numpy as np
from typing import Tuple, Optional

# Configure logger
logger = logging.getLogger(__name__)

class MecanumBase:
    """
    Driver for a custom ESP32-based Mecanum base using a string-based protocol.
    
    Protocol:
        Input (Jetson -> ESP32): "TWIST,vx,vy,omega\n"
        Output (ESP32 -> Jetson): "ODOM,x,y,theta,vx,vy,omega,enc1,enc2,enc3,enc4\n"
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout: float = 0.05):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        
        # State storage [x, y, theta]
        self._latest_odom = np.zeros(3, dtype=np.float32)
        # Velocity storage [vx, vy, omega] (read back from base for confirmation)
        self._latest_vel = np.zeros(3, dtype=np.float32)

    def connect(self):
        """Establishes the serial connection."""
        if self.ser is None or not self.ser.is_open:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                # clear buffers to remove old data
                self.ser.reset_input_buffer()
                logger.info(f"Connected to Mecanum Base on {self.port}")
            except serial.SerialException as e:
                logger.error(f"Failed to connect to Mecanum Base on {self.port}: {e}")
                raise e

    def disconnect(self):
        """Closes the serial connection."""
        if self.ser and self.ser.is_open:
            # Send stop command before closing
            self.send_twist(0.0, 0.0, 0.0)
            self.ser.close()
            logger.info("Disconnected from Mecanum Base")

    def send_twist(self, vx: float, vy: float, omega: float):
        """
        Sends velocity commands to the base.
        args:
            vx: Linear velocity X (forward) in m/s
            vy: Linear velocity Y (left) in m/s
            omega: Angular velocity Z (CCW) in rad/s
        """
        if not self.ser or not self.ser.is_open:
            logger.warning("Attempted to write to closed Mecanum Base.")
            return

        # Format: TWIST,linear_x,linear_y,angular_z
        # Using .3f to keep string length reasonable while maintaining precision
        msg = f"TWIST,{vx:.3f},{vy:.3f},{omega:.3f}\n"
        
        try:
            self.ser.write(msg.encode('utf-8'))
        except serial.SerialException as e:
            logger.error(f"Serial write failed: {e}")

    def read_odom(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads the latest line from the serial buffer. 
        Note: This clears the buffer to ensure we get the MOST RECENT packet.
        
        Returns:
            odom (np.array): [x, y, theta]
            vel (np.array): [vx, vy, omega]
        """
        if not self.ser or not self.ser.is_open:
            return self._latest_odom, self._latest_vel

        line = None
        try:
            # Loop to read everything currently in the buffer. 
            # We only care about the very last complete line (lowest latency).
            while self.ser.in_waiting > 0:
                raw_line = self.ser.readline()
                # Basic check to see if line looks complete (ends in newline)
                if raw_line.endswith(b'\n'):
                    line = raw_line

            if line:
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                self._parse_odom_string(decoded_line)

        except Exception as e:
            logger.warning(f"Error reading/parsing ODOM: {e}")

        return self._latest_odom, self._latest_vel

    def _parse_odom_string(self, data_str: str):
        """
        Parses string: "ODOM,x,y,theta,vx,vy,omega,enc1,enc2,enc3,enc4"
        """
        if not data_str.startswith("ODOM"):
            return

        try:
            parts = data_str.split(',')
            if len(parts) < 7: # We need at least up to omega
                return

            # Indices: 0=Header, 1=x, 2=y, 3=theta, 4=vx, 5=vy, 6=omega
            x = float(parts[1])
            y = float(parts[2])
            theta = float(parts[3])
            
            vx = float(parts[4])
            vy = float(parts[5])
            omega = float(parts[6])

            # Update internal state
            self._latest_odom[:] = [x, y, theta]
            self._latest_vel[:] = [vx, vy, omega]
            
        except ValueError:
            pass # Malformed number in string

    def __del__(self):
        self.disconnect()