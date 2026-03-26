import time
import threading
import torch 
from lerobot.robots.mimic_follower import MimicFollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Updated import for XVLA (assuming standard LeRobot fork structure)
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

# Shared state protected by a lock (mutex)
current_task = "wait"
task_lock = threading.Lock()

def brain_thread():
    """This thread handles the LLM, solver, and updating the task."""
    global current_task
    while True:
        move = input("\nEnter grid position (1-18) or 'wait': ")
        
        if move in TIC_TAC_TOE_MOVES:
            with task_lock:
                current_task = TIC_TAC_TOE_MOVES[move]
            print(f"> Instructing XVLA: {current_task}")
        elif move == "wait":
            with task_lock:
                current_task = "wait"
            print("> Pausing robot.")

def main():
    print("Initializing robot configuration...")
    # Updated to ONLY use the 2 cameras specified in your lerobot-record bash script
    robot_cfg = MimicFollowerConfig(
        id="mimic_follower", 
        left_arm_port="/dev/arm_left_follower",
        right_arm_port="/dev/arm_right_follower",
        base_port="/dev/mecanum_base",
        cameras={
            "image": OpenCVCameraConfig(index_or_path="/dev/camera_left_wrist", width=640, height=480, fps=30, fourcc="MJPG", warmup_s=0),
            "image2": OpenCVCameraConfig(index_or_path="/dev/camera_right_wrist", width=640, height=480, fps=30, fourcc="MJPG", warmup_s=0)
        }
    )
    robot = make_robot_from_config(robot_cfg)
    
    print("Loading XVLA weights into the GPU...")
    # Updated to your specific XVLA model checkpoint
    # model_id = "Mimic-Robotics/xvla_full_ttt_18e_300k"
    # model_id = "Mimic-Robotics/xvla_odin_ttt__xvla_nofr_2cam_30hz_300k"
    # Mimic-Robotics/xvla_odin_ttt__xvla_nofr_2cam_30hz_300k
    model_id = "Mimic-Robotics/xvla_speed_ttt_3cam_15hz_32ac_b40_21_mar_allckpt"
    # Load the policy directly from the pretrained checkpoint
    policy = XVLAPolicy.from_pretrained(model_id)
    policy.to("cuda")
    policy.eval() # Ensure the model is locked into inference mode
    
    # Load the processors using the stats saved in your Hugging Face repo
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=model_id,
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
    
    print("Connecting to robot...")
    robot.connect(calibrate=False)
    
    # Start the background thread for the brain
    threading.Thread(target=brain_thread, daemon=True).start()

    print("Robot ready! Starting control loop.")
    
    # The Continuous Control Loop
    while True:
        start_t = time.perf_counter()
        
        # Safely read the current instruction
        with task_lock:
            active_task = current_task
            
        if active_task == "wait":
            time.sleep(1/30) # Maintain 30fps idle
            continue
            
        # Get live data
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        
        # Native LeRobot packaging (Drops floats, formats NumPy arrays, maps prefixes)
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
        
        # Convert the raw PyTorch tensor into the formatted bimanual RobotAction dictionary
        act_processed_policy = make_robot_action(action_values, dataset_features)
        
        # Execute
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        robot.send_action(robot_action_to_send)
        
        # Keep the loop steady at ~30Hz
        time.sleep(max(1/30 - (time.perf_counter() - start_t), 0))

if __name__ == "__main__":
    main()