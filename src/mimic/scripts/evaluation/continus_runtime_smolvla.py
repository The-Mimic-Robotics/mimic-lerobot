    # model_id = "Mimic-Robotics/smol_full_ttt_70k"
    
    
    # model_id = "Mimic-Robotics/smol_redx_nofr_15hz_32ac_3cam_75k"
    # model_id = "Mimic-Robotics/smol_StableRed_15hz_32ac_uf_40k"
    
import time
import threading
import torch
import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.robots.mimic_follower import MimicFollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.zed_camera import ZedCameraConfig

# Teleoperation imports
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.mimic_leader import MimicLeaderConfig

# SmolVLA imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.utils.control_utils import predict_action

from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.constants import OBS_STR
from lerobot.policies.utils import make_robot_action
from lerobot.datasets.utils import combine_feature_dicts

# ==========================================
# ⚙️ QUICK CONFIG
# ==========================================
# Easily swap your model checkpoint here
MODEL_ID = "Mimic-Robotics/smol_StableRed_15hz_32ac_uf_40k"
DENOISE_STEPS = 150

# Target FPS for the policy execution
CONTROL_HZ = 15
# ==========================================

# 1. Map your grid inputs (1-18) to the exact language commands Smol knows
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

# Shared state protected by a lock (mutex)
current_task = "wait"
task_lock = threading.Lock()

def brain_thread():
    """This thread handles user input and updates the active task."""
    global current_task
    while True:
        print("\nCommands:")
        print("  [1-18] Tic-Tac-Toe grid position")
        print("  [20]   Custom prompt")
        print("  [-1]   Teleoperation (Human Control)")
        print("  [0]    Pause robot")
        move = input("Enter command: ").strip()
        
        if move == "-1":
            with task_lock:
                current_task = "teleop"
            print("> Switched to Teleoperation Mode. Follower mirroring leader arms.")
            
        elif move == "20":
            custom_prompt = input("Enter custom instruction: ").strip()
            with task_lock:
                current_task = custom_prompt
            print(f"> Instructing Smol-VLA: {current_task}")
            
        elif move in TIC_TAC_TOE_MOVES:
            with task_lock:
                current_task = TIC_TAC_TOE_MOVES[move]
            print(f"> Instructing Smol-VLA: {current_task}")
            
        elif move == "wait" or move == "0":
            with task_lock:
                current_task = "wait"
            print("> Pausing robot.")
            
        else:
            print("> Unknown command. Please try again.")


def main():
    # ---------------------------------------------------------
    # 2. Configuration & Initialization
    # ---------------------------------------------------------
    print("Configuring Follower Robot...")
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
    
    print("Configuring Leader Arms (Teleop)...")
    teleop_cfg = MimicLeaderConfig(
        id="mimic_leader",
        left_arm_port="/dev/arm_left_leader",
        right_arm_port="/dev/arm_right_leader",
        base_control_mode="keyboard"
    )
    teleop = make_teleoperator_from_config(teleop_cfg)

    print(f"Loading SmolVLA weights ({MODEL_ID}) into the GPU...")
    
    # Load the policy directly from the pretrained checkpoint
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.to("cuda", dtype=torch.bfloat16) # Added bfloat16 casting
    
    # Optional config overrides mirrored from XVLA
    if hasattr(policy.config, 'num_denoising_steps'):
        policy.config.num_denoising_steps = DENOISE_STEPS
    policy.config.use_amp = True
    
    policy.eval() # Ensure the model is locked into inference mode
    
    # Load the processors using the stats saved in your Hugging Face repo
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": "cuda"}}
    )
    
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # Create the complete feature schema including BOTH actions and observations
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
    
    # ---------------------------------------------------------
    # 3. Connection & Calibration
    # ---------------------------------------------------------
    print("Connecting to Follower...")
    robot.connect(calibrate=False)
    
    print("Connecting to Leader...")
    teleop.connect()
    
    # Start the background thread for the brain
    threading.Thread(target=brain_thread, daemon=True).start()

    print(f"\n✅ Ready! Starting control loop at {CONTROL_HZ}Hz. Press Ctrl+C to safely exit.")
    
    # ---------------------------------------------------------
    # 4. The Control Loop
    # ---------------------------------------------------------
    try:
        while True:
            start_t = time.perf_counter()
            
            # Safely read the current instruction
            with task_lock:
                active_task = current_task
                
            if active_task == "wait":
                time.sleep(1 / CONTROL_HZ)
                continue
                
            # Get live data
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            
            # --- TELEOPERATION MODE ---
            if active_task == "teleop":
                act = teleop.get_action()
                act_processed_teleop = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
                robot.send_action(robot_action_to_send)
                
            # --- AI POLICY MODE ---
            else:
                # Native LeRobot packaging
                observation_frame = build_dataset_frame(obs_features, obs_processed, prefix=OBS_STR)
                
                # Predict the action based on the LIVE text string!
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
                
                # Convert raw PyTorch tensor into formatted RobotAction dict
                act_processed_policy = make_robot_action(action_values, dataset_features)
                robot_action_to_send = robot_action_processor((act_processed_policy, obs))
                robot.send_action(robot_action_to_send)
            
            # Keep the loop steady
            dt_s = time.perf_counter() - start_t
            time.sleep(max(0, (1 / CONTROL_HZ) - dt_s))
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping control loop...")
        
    finally:
        # ---------------------------------------------------------
        # 5. Safe Shutdown
        # ---------------------------------------------------------
        print("Disconnecting arms and disabling torque safely...")
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()