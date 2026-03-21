#!/bin/bash
# Batch convert all old bimanual handover datasets to mobile bimanual format
# Uses convert_bimanual_complete.py which properly renames cameras and adds front camera

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of all old datasets (from HuggingFace)
OLD_DATASETS=(
    "bimanual_blue_block_handover_1"
    "bimanual_blue_block_handover_2"
    "bimanual_blue_block_handover_3"
    "bimanual_blue_block_handover_4"
    "bimanual_blue_block_handover_5"
    "bimanual_blue_block_handover_6"
    "bimanual_blue_block_handover_7"
    "bimanual_blue_block_handover_14"
    "bimanual_blue_block_handover_15"
    "bimanual_blue_block_handover_16"
    "bimanual_blue_block_handover_17"
    "bimanual_blue_block_handover_18"
    "bimanual_blue_block_handover_19"
    "bimanual_blue_block_handover_20"
    "bimanual_blue_block_handover_21"
    "bimanual_blue_block_handover_22"
    "bimanual_blue_block_handover_23"
    "bimanual_blue_block_handover_24"
    "bimanual_blue_block_handover_25"
    "bimanual_blue_block_handover_26"
)

ORG="Mimic-Robotics"

echo "========================================"
echo "Batch Dataset Conversion"
echo "========================================"
echo "Total datasets to convert: ${#OLD_DATASETS[@]}"
echo "Using: convert_bimanual_complete.py"
echo ""

for dataset in "${OLD_DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Converting: $dataset"
    echo "----------------------------------------"
    
    INPUT_REPO="${ORG}/${dataset}"
    OUTPUT_REPO="${ORG}/mobile_${dataset}"
    
    python3 "${SCRIPT_DIR}/convert_bimanual_complete.py" \
        --input-repo="$INPUT_REPO" \
        --output-repo="$OUTPUT_REPO"
    
    if [ $? -eq 0 ]; then
        echo "Successfully converted $dataset"
    else
        echo "Failed to convert $dataset"
        exit 1
    fi
    
    echo ""
done

echo "========================================"
echo "All datasets converted successfully!"
echo "========================================"
