The Strategy: "Zero-Padding" & Key Remapping

Since your new action space includes the old one (it is a superset), we can mathematically transform the old data to look like the new data.

    Action Space:

        Old: [Left_Arm (6), Right_Arm (6)]

        New: [Left_Arm (6), Right_Arm (6), Base (3)]

        The Fix: Append [0.0, 0.0, 0.0] to every action in the old dataset. This teaches the robot: "When doing these specific manipulation tasks, keep the base still."

    Observation Space:

        Old: [Left_Arm_State (6), Right_Arm_State (6)]

        New: [Left_Arm_State (6), Right_Arm_State (6), Base_Odom (3)]

        The Fix: Append [0.0, 0.0, 0.0] (or valid placeholders) to the state.

    Cameras (The Tricky Part):

        Renaming: You must rename the keys in the dataset so the training script treats them as the same cameras.

            wrist_right → right_wrist

            wrist_left → left_wrist

            realsense_top → head

        The Risk (Extrinsics): If your ZED camera is in a completely different spot than the RealSense was (e.g., 20cm higher), the model might struggle. If they are in roughly the same "head" position, a modern Vision Transformer (like in Pi0 or ACT) can often handle the domain shift, especially if you fine-tune.


        possible code ~~~ => import torch
import numpy as np
from datasets import load_dataset, Features, Sequence, Value, Array3D, Array2D

# --- CONFIGURATION ---
SOURCE_REPO = "your_username/old_bimanual_dataset"  # Your old 450 ep dataset
TARGET_REPO = "your_username/mimic_combined_dataset" # The new generic home
HF_TOKEN = None # Set this if you haven't logged in via CLI

# Define the features we want to KEEP and RENAME
# Old Key -> New Key
KEY_MAPPING = {
    "observation.images.wrist_right": "observation.images.right_wrist",
    "observation.images.wrist_left":  "observation.images.left_wrist",
    "observation.images.realsense_top": "observation.images.head",
    "observation.state": "observation.state",
    "action": "action",
    # Keep other metadata
    "episode_index": "episode_index",
    "frame_index": "frame_index",
    "timestamp": "timestamp",
    "next.done": "next.done",
}

def transform_batch(batch):
    new_batch = {new_k: [] for new_k in KEY_MAPPING.values()}
    
    batch_size = len(batch["action"])
    
    for i in range(batch_size):
        # 1. IMAGES & STATE
        for old_k, new_k in KEY_MAPPING.items():
            if old_k in batch:
                new_batch[new_k].append(batch[old_k][i])
        
        # 2. ACTION PADDING (12 -> 15)
        # Old: 12 arms
        # New: 12 arms + 3 base (0,0,0)
        old_action = np.array(batch["action"][i], dtype=np.float32)
        base_zeros = np.zeros(3, dtype=np.float32)
        new_action = np.concatenate([old_action, base_zeros])
        new_batch["action"][-1] = new_action # Update the last added item

        # 3. OBSERVATION STATE PADDING (12 -> 15)
        # Old: 12 motor pos
        # New: 12 motor pos + 3 base odom (0,0,0)
        old_state = np.array(batch["observation.state"][i], dtype=np.float32)
        new_state = np.concatenate([old_state, base_zeros])
        new_batch["observation.state"][-1] = new_state

    return new_batch

def main():
    print(f"Loading {SOURCE_REPO}...")
    ds = load_dataset(SOURCE_REPO, split="train")
    
    print("Transforming data...")
    # remove_columns deletes old keys that aren't in our new mapping
    new_ds = ds.map(
        transform_batch, 
        batched=True, 
        batch_size=100,
        remove_columns=ds.column_names
    )

    # Cast features if necessary (LeRobot likes strict typing)
    # usually datasets infers this, but being safe helps
    
    print(f"Pushing to {TARGET_REPO}...")
    new_ds.push_to_hub(TARGET_REPO, token=HF_TOKEN)
    print("Done! You can now train on this mixed with your new data.")

if __name__ == "__main__":
    main()