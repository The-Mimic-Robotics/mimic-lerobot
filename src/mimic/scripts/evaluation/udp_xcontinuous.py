import time
import threading
import torch
import numpy as np
import socket

from lerobot.robots import make_robot_from_config
from lerobot.robots.mimic_follower import MimicFollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.zed_camera import ZedCameraConfig

# Updated import for XVLA
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.utils.control_utils import predict_action

from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.constants import OBS_STR

from lerobot.policies.utils import make_robot_action
from lerobot.datasets.utils import combine_feature_dicts

# 1. Map your grid inputs (1-18) to the exact language commands XVLA knows
TIC_TAC_TOE_MOVES = {
    "0": "wait",
    "1": "pick red x piece handover place bottom left",
    "2": "pick red x piece handover place bottom middle",
    "3": "pick red x piece handover place bottom right",
    "4": "pick red x handover place middle left",
    "5": "pick red x handover place center",
    "6": "pick red x handover place middle right",
    "7": "pick red x piece handover place top left",
    "8": "pick red x piece handover place top middle",
    "9": "pick red x piece handover place top right",
    "10": "pick blue o piece handover place bottom left",
    "11": "pick blue o piece handover place bottom middle",
    "12": "pick blue o piece handover place bottom right",
    "13": "pick blue o handover place middle left",
    "14": "pick blue o handover place center",
    "15": "pick blue o handover place middle right",
    "16": "pick blue o piece handover place top left",
    "17": "pick blue o piece handover place top middle",
    "18": "pick blue o piece handover place top right"
}

# --- UDP Configuration ---
UDP_IP = "0.0.0.0" 
UDP_PORT = 5005

# Shared state protected by a lock
current_task = "wait"
task_lock = threading.Lock()

def udp_listener_thread():
    """Listens for incoming UDP IDs and updates the task for XVLA."""
    global current_task
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"\n[Network] XVLA listening for UDP commands on port {UDP_PORT}...")
    
    while True:
        data, addr = sock.recvfrom(1024) 
        move = data.decode('utf-8').strip()
        
        if move in TIC_TAC_TOE_MOVES:
            with task_lock:
                current_task = TIC_TAC_TOE_MOVES[move]
            print(f"\n> [UDP] Received Move ID {move}. Instructing XVLA: {current_task}")
        elif move == "0" or move.lower() == "wait":
            with task_lock:
                current_task = "wait"
            print("\n> [UDP] Received WAIT command. Pausing robot.")
        else:
            print(f"\n> [UDP] Ignored unknown command: {move}")

def main():
    print("Initializing robot configuration...")
    # Using the same 3-camera config as your smolVLA script for consistency
    robot_cfg = MimicFollowerConfig(
        id="mimic_follower", 
        left_arm_port="/dev/arm_left_follower",
        right_arm_port="/dev/arm_right_follower",
        base_port="/dev/mecanum_base",
        cameras={
            "right_wrist": OpenCVCameraConfig(index_or_path="/dev/camera_right_wrist", width=640, height=480, fps=30, fourcc="MJPG", warmup_s=0),
            "left_wrist": OpenCVCameraConfig(index_or_path="/dev/camera_left_wrist", width=640, height=480, fps=30, fourcc="MJPG", warmup_s=0),
            "top": ZedCameraConfig(index_or_path="23081456", width=1280, height=720, fps=30, warmup_s=0)
        }
    )
    robot = make_robot_from_config(robot_cfg)
    
    print("Loading XVLA weights into the GPU...")
    model_id = "Mimic-Robotics/xvla_ttt_15hz_32ac_iTT_200k"
    
    # Load the XVLA policy
    policy = XVLAPolicy.from_pretrained(model_id)
    policy.to("cuda")
    policy.eval() 
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=model_id,
        preprocessor_overrides={"device_processor": {"device": "cuda"}}
    )
    
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # Setup feature schemas
    action_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=False
    )
    obs_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=True 
    )
    dataset_features = combine_feature_dicts(action_features, obs_features)
    
    print("Connecting to robot...")
    robot.connect(calibrate=False)
    
    # Start the UDP listener in the background
    threading.Thread(target=udp_listener_thread, daemon=True).start()

    print("XVLA Robot ready! Starting control loop.")
    
    while True:
        start_t = time.perf_counter()
        
        with task_lock:
            active_task = current_task
            
        if active_task == "wait":
            time.sleep(1/30)
            continue
            
        # Get live data
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(obs_features, obs_processed, prefix=OBS_STR)
        
        # Inference
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=torch.device("cuda"),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=active_task,
            robot_type=robot.robot_type
        )
        
        # Process and send action
        act_processed_policy = make_robot_action(action_values, dataset_features)
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        robot.send_action(robot_action_to_send)
        
        # Maintain 30Hz
        time.sleep(max(1/30 - (time.perf_counter() - start_t), 0))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nXVLA shutdown.")