
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

def inspect_dataset(repo_id):
    print(f"--- Inspecting {repo_id} ---")
    meta = LeRobotDatasetMetadata(repo_id)
    print(f"Features: {list(meta.features.keys())}")
    
    if "action" in meta.features:
        print(f"Action shape: {meta.features['action']['shape']}")
    if "observation.state" in meta.features:
        print(f"Observation State shape: {meta.features['observation.state']['shape']}")
    print("\n")

# Old dataset (Stationary bimanual)
inspect_dataset("Mimic-Robotics/bimanual_blue_block_handover_1")

# New dataset (Mobile bimanual)
inspect_dataset("Mimic-Robotics/mobile_bimanual_blue_block_handover_1")
