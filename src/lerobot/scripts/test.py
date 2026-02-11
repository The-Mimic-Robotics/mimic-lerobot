from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# Point to your LOCAL merged folder
LOCAL_DIR = Path("data/merged_tictactoe_10hz")
dataset = LeRobotDataset(LOCAL_DIR)

# Get all unique task indices
unique_tasks = set(dataset.hf_dataset["task_index"])
print(f"Found Task IDs: {unique_tasks}")

# Check the task descriptions
print("\nTask Definitions (from meta/tasks.jsonl):")
# Note: In newer LeRobot versions, this might be accessed differently, 
# but usually it's in dataset.meta.tasks or we read the file directly.
try:
    print(dataset.meta.tasks)
except:
    # Fallback: Read the file manually
    import json
    task_file = LOCAL_DIR / "meta/tasks.jsonl"
    if task_file.exists():
        with open(task_file, 'r') as f:
            for line in f:
                print(json.loads(line))
    else:
        print("No tasks.jsonl file found!")