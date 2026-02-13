from huggingface_hub import HfApi

# Your full list of datasets to fix
dataset_repos = [
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_center_slow_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_center_slow_v2",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v2",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v3",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v4",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v5",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v6",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v8",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v9",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v11",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v12",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v13",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v14",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v15",
    "Mimic-Robotics/mimic_tictactoe_pick_red_x_handover_place_mm_v16",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topleft_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topleft_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_toprght_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_toprght_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_toprght_v3",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_toprght_v4",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottomleft_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottomleft_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottomright_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottomright_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topmiddle_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topMiddle_v3",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topMiddle_v4",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v3",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v4",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleRight_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleRight_v2",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottom_middle_v1",
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottom_middle_v2"
]

api = HfApi()

print("Applying v3.0 tags to datasets...")

for repo_id in dataset_repos:
    try:
        # 1. First we try to read info.json to see what version it CLAIMS to be
        # (This is just a check, but usually it's safest to just force v3.0 if you are using latest lerobot)
        
        # 2. Force create the tag "v3.0"
        # If your data is older (v2.1), change this to "v2.1"
        target_tag = "v3.0" 
        
        api.create_tag(
            repo_id=repo_id,
            tag=target_tag,
            repo_type="dataset",
            exist_ok=True  # Don't crash if tag already exists
        )
        print(f"✅ Tagged {repo_id} with {target_tag}")
        
    except Exception as e:
        print(f"❌ Failed to tag {repo_id}: {e}")

print("Done! You can now run the merge command.")