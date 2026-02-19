import time
import threading
from lerobot.robots import make_robot_from_config, RobotConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.utils.control_utils import predict_action

# 1. Map your grid inputs (1-9) to the exact language commands X-VLA knows
TIC_TAC_TOE_MOVES = {
    "1": "pick red x piece handover place top left",
    "2": "pick red x piece handover place top center",
    "3": "pick red x piece handover place top right",
    # ... fill in 4 through 9
}

# Shared state protected by a lock (mutex)
current_task = "wait"
task_lock = threading.Lock()

def brain_thread():
    """This thread handles the LLM, solver, and updating the task."""
    global current_task
    while True:
        # Here is where your LLM/Solver logic goes. 
        # For testing, we just use a simple input prompt:
        move = input("\nEnter grid position (1-9) or 'wait': ")
        
        if move in TIC_TAC_TOE_MOVES:
            with task_lock:
                current_task = TIC_TAC_TOE_MOVES[move]
            print(f"> Instructing X-VLA: {current_task}")
        elif move == "wait":
            with task_lock:
                current_task = "wait"
            print("> Pausing robot.")

def main():
    # 2. Initialization: This is where the 20-second wait happens (ONCE)
    print("Loading smolVLA weights into the 5070...")
    
    # Configure your mimic_follower robot (mimicking your bash script parameters)
    robot_cfg = RobotConfig(
        type="mimic_follower",
        left_arm_port="/dev/arm_left_follower",
        right_arm_port="/dev/arm_right_follower",
        base_port="/dev/mecanum_base",
        cameras={
            "right_wrist": {"type": "opencv", "index_or_path": "/dev/camera_right_wrist", "width": 640, "height": 480, "fps": 30},
            "left_wrist": {"type": "opencv", "index_or_path": "/dev/camera_left_wrist", "width": 640, "height": 480, "fps": 30},
            "top": {"type": "zed_camera", "index_or_path": "23081456", "width": 1280, "height": 720, "fps": 30}
        }
    )
    robot = make_robot_from_config(robot_cfg)
    
    policy_cfg = PreTrainedConfig.from_pretrained("Mimic-Robotics/smol_augusto_redxVlm_50a_48b_100k")
    policy = make_policy(policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg)
    
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    
    robot.connect()
    
    # 3. Start the background thread for the brain
    threading.Thread(target=brain_thread, daemon=True).start()

    print("Robot ready! Starting control loop.")
    
    # 4. The Continuous Control Loop
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
        
        # Predict the action based on the LIVE text string!
        action_values = predict_action(
            observation=obs_processed,
            policy=policy,
            device="cuda",
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=active_task,
            robot_type=robot.robot_type
        )
        
        # Execute
        robot_action_to_send = robot_action_processor((action_values, obs))
        robot.send_action(robot_action_to_send)
        
        # Keep the loop steady at ~30Hz
        time.sleep(max(1/30 - (time.perf_counter() - start_t), 0))

if __name__ == "__main__":
    main()