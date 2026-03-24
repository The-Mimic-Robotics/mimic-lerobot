import time
import threading

from lerobot.robots import make_robot_from_config
from lerobot.robots.mimic_follower import MimicFollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.zed_camera import ZedCameraConfig

# Teleoperation imports
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.mimic_leader import MimicLeaderConfig

# Dataset and Processor imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors

# Shared state protected by a lock (mutex)
current_task = "wait"
task_lock = threading.Lock()

# Note: Changed to 30 Hz because your dataset is named "30hz". 
# Running a 30hz dataset at 15hz would cause it to play in slow-motion!
CONTROL_HZ = 60 

def brain_thread():
    """This thread handles user input and updates the active task."""
    global current_task
    while True:
        print("\nCommands:")
        print(" [0-8] Play Tic-Tac-Toe placement episode (0 to 8)")
        print(" [-1]  Teleoperation (Human Control)")
        print(" [w]   Wait / Pause robot")
        move = input("Enter command: ").strip()
        
        if move == "-1":
            with task_lock:
                current_task = "teleop"
            print("> Switched to Teleoperation Mode. Follower mirroring leader arms.")
            
        elif move in [str(i) for i in range(9)]:
            with task_lock:
                current_task = f"replay_{move}"
            print(f"> Replaying episode {move}...")
            
        elif move.lower() == "w" or move == "wait":
            with task_lock:
                current_task = "wait"
            print("> Pausing robot.")
            
        else:
            print("> Unknown command. Please try again.")

def main():
    global current_task
    # ---------------------------------------------------------
    # 1. Configuration & Initialization
    # ---------------------------------------------------------
    print("Configuring Follower Robot...")
    robot_cfg = MimicFollowerConfig(
        id="mimic_follower", 
        left_arm_port="/dev/arm_left_follower",
        right_arm_port="/dev/arm_right_follower",
        base_port="/dev/mecanum_base",
        cameras={
            "image": ZedCameraConfig(index_or_path="23081456", width=1280, height=720, fps=30, warmup_s=0, use_depth=False),
            "image2": OpenCVCameraConfig(index_or_path="/dev/camera_left_wrist", width=640, height=480, fps=30, fourcc="MJPG", warmup_s=0),
            "image3": OpenCVCameraConfig(index_or_path="/dev/camera_right_wrist", width=640, height=480, fps=30, fourcc="MJPG", warmup_s=0)
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

    teleop_action_processor, robot_action_processor, _ = make_default_processors()
    
    # ---------------------------------------------------------
    # 2. Dataset Loading & Caching
    # ---------------------------------------------------------
    print("Loading Dataset and Pre-caching episodes 0-8 to memory...")
    dataset_id = "Mimic-Robotics/mimic_ttt_redx_30hz_hardcoded_placement"
    dataset = LeRobotDataset(dataset_id)
    
    episodes_cache = {}
    for i in range(9):
        # Filter dataset frames by episode index and isolate the actions
        ep_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == i)
        episodes_cache[i] = ep_frames.select_columns("action")
    print("Caching complete!")

    # ---------------------------------------------------------
    # 3. Connection & Calibration
    # ---------------------------------------------------------
    print("Connecting to Follower...")
    robot.connect(calibrate=False)
    
    print("Connecting to Leader...")
    teleop.connect()
    
    threading.Thread(target=brain_thread, daemon=True).start()

    print(f"\n✅ Ready! Starting control loop at {CONTROL_HZ}Hz. Press Ctrl+C to safely exit.")
    
    # State tracking variables for the replay loop
    active_episode = None
    episode_step = 0
    
    # ---------------------------------------------------------
    # 4. The Control Loop
    # ---------------------------------------------------------
    try:
        while True:
            start_t = time.perf_counter()
            
            with task_lock:
                active_task = current_task
                
            if active_task == "wait":
                active_episode = None # Reset replay state if we paused
                time.sleep(1 / CONTROL_HZ)
                continue
                
            obs = robot.get_observation()
            
            # --- REPLAY MODE ---
            if active_task.startswith("replay_"):
                ep_idx = int(active_task.split("_")[1])
                
                # If we just switched to a new episode, reset the frame counter
                if active_episode != ep_idx:
                    active_episode = ep_idx
                    episode_step = 0
                
                # Play the frame if we haven't reached the end of the episode
                if episode_step < len(episodes_cache[ep_idx]):
                    action_array = episodes_cache[ep_idx][episode_step]["action"]
                    
                    # Convert raw array back into the dictionary LeRobot expects
                    action_dict = {}
                    for i, name in enumerate(dataset.features["action"]["names"]):
                        action_dict[name] = action_array[i]
                        
                    processed_action = robot_action_processor((action_dict, obs))
                    robot.send_action(processed_action)
                    
                    episode_step += 1
                else:
                    # Episode is finished
                    print(f"> Episode {ep_idx} complete. Returning to wait state.")
                    with task_lock:
                        # Only reset to wait if the user hasn't typed a new command in the meantime
                        if current_task == active_task:
                            current_task = "wait"
                    active_episode = None
                
            # --- TELEOPERATION MODE ---
            elif active_task == "teleop":
                active_episode = None # Reset replay state
                act = teleop.get_action()
                act_processed_teleop = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
                robot.send_action(robot_action_to_send)
            
            # Keep loop steady
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