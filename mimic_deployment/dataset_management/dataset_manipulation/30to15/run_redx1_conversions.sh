#!/bin/bash

# nohup ./run_redx1_conversions.sh > redx1_conversion.log 2>&1 &
# Define the exact mapping for each dataset.
# Format: "Original_Repo_ID|Destination_Repo_ID"
mappings=(
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_center_slow_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_center_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_center_slow_v2|Mimic-Robotics/mimic_ttt_redx_15hz_x1_center_BGR_v2"
    
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topleft_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TL_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topleft_v2|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TL_BGR_v2"
    
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_toprght_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TR_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_toprght_v2|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TR_BGR_v2"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_toprght_v3|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TR_BGR_v3"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_toprght_v4|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TR_BGR_v4"
    
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottomleft_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_BL_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottomleft_v2|Mimic-Robotics/mimic_ttt_redx_15hz_x1_BL_BGR_v2"
    
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottomright_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_BR_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottomright_v2|Mimic-Robotics/mimic_ttt_redx_15hz_x1_BR_BGR_v2"
    
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topmiddle_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TM_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topMiddle_v3|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TM_BGR_v3"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_topMiddle_v4|Mimic-Robotics/mimic_ttt_redx_15hz_x1_TM_BGR_v4"
    
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_ML_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v2|Mimic-Robotics/mimic_ttt_redx_15hz_x1_ML_BGR_v2"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v3|Mimic-Robotics/mimic_ttt_redx_15hz_x1_ML_BGR_v3"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleLeft_v4|Mimic-Robotics/mimic_ttt_redx_15hz_x1_ML_BGR_v4"
    
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleRight_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_MR_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_middleRight_v2|Mimic-Robotics/mimic_ttt_redx_15hz_x1_MR_BGR_v2"
    
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottom_middle_v1|Mimic-Robotics/mimic_ttt_redx_15hz_x1_BM_BGR_v1"
    "Mimic-Robotics/mimic_tictactoe_red_x_handover_bottom_middle_v2|Mimic-Robotics/mimic_ttt_redx_15hz_x1_BM_BGR_v2"
)

echo "Starting batch conversion of ${#mappings[@]} datasets..."

# Loop through each mapping
for mapping in "${mappings[@]}"; do
    # Split the string at the pipe character "|"
    orig="${mapping%%|*}"
    dest="${mapping##*|}"
    
    echo "======================================================"
    echo "Processing: $orig"
    echo "Target:     $dest"
    echo "======================================================"
    
    # Call the python script with the arguments
    python 30to15.py --orig "$orig" --dest "$dest"
    
    # Optional: Add a small delay between runs to let memory clear completely
    sleep 5
done

echo "ALL RED_X1 CONVERSIONS COMPLETE!"