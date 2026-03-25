#!/bin/bash
#nohup ./run_conversions.sh > master_conversion.log 2>&1 &
# Define the array of original datasets
datasets=(
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BL_v2"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BL_v3"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BM_v2"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BM_v3"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BR_v2"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_BR_v3"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TL_v1"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TL_v2"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TL_v3"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_ML_v2"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_ML_v3"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_center_v1"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_MR_v1"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_MR_v2"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TM_v1"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TM_v2"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TR_v2"
    "Mimic-Robotics/mimic_ttt_redx_30hz_x2_TR_v3"
)

echo "Starting batch conversion of ${#datasets[@]} datasets..."

# Loop through each dataset one by one
for orig in "${datasets[@]}"; do
    # String manipulation: replace "30hz" with "15hz"
    dest="${orig/30hz/15hz}"
    
    echo "======================================================"
    echo "Processing: $orig"
    echo "Target:     $dest"
    echo "======================================================"
    
    # Call the python script with the arguments
    python 30to15.py --orig "$orig" --dest "$dest"
    
    # Optional: Add a small delay between runs to let memory clear completely
    sleep 5
done

echo "ALL CONVERSIONS COMPLETE!"