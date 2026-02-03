#!/bin/bash
# Master Training Orchestrator for Mimic Robot
# Manages multi-policy training across multiple computers with dataset grouping

set -e

# ============================================================================
# SCRIPT CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET_RESOLVER="$SCRIPT_DIR/dataset_groups.py"

# ============================================================================
# COLOR OUTPUT
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# USAGE
# ============================================================================

usage() {
    cat << EOF
${BLUE}═══════════════════════════════════════════════════════════════════${NC}
${GREEN}Mimic Robot Training Manager${NC}
${BLUE}═══════════════════════════════════════════════════════════════════${NC}

Usage: $0 [OPTIONS]

${YELLOW}Required Options:${NC}
  --policy POLICY[,POLICY2,...]  Policy type(s) to train (comma-separated for multiple)
                                  Choices: xvla, pi05, pi0, groot, act, diffusion
                                  Example: --policy xvla,pi05 (trains both sequentially)
  
  --dataset-group GROUP           Dataset group to train on
                                  Run with --list-groups to see available groups
  
  --dataset REPO_ID               Single dataset repository ID (alternative to --dataset-group)
                                  Example: --dataset Mimic-Robotics/dataset_name

${YELLOW}Optional:${NC}
  --computer NAME          Computer name (default: auto-detect from hostname)
  --batch-size SIZE        Override default batch size
  --num-workers N          Override default number of workers
  --steps N                Number of training steps
  --list-groups            List all available dataset groups and exit
  --list-policies          List all available policies and exit
  --dry-run                Show configuration without starting training
  -h, --help               Show this help message

${YELLOW}Environment Variables:${NC}
  COMPUTER                 Same as --computer flag
  DATASET_GROUP            Same as --dataset-group flag
  BATCH_SIZE               Same as --batch-size flag
  NUM_WORKERS              Same as --num-workers flag
  STEPS                    Same as --steps flag

${YELLOW}Examples:${NC}
  # Train xVLA on all datasets
  $0 --policy xvla --dataset-group all_datasets

  # Train multiple policies sequentially on same dataset group
  $0 --policy xvla,pi05 --dataset-group high_quality

  # Train on a single dataset
  $0 --policy act --dataset Mimic-Robotics/mimic_displacement_to_handover_blue_block_with_new_hands_v3

  # Train ACT on stationary datasets on jupiter computer
  $0 --policy act --dataset-group stationary --computer jupiter

  # Dry run to see configuration
  $0 --policy groot --dataset-group most_recent --dry-run

  # List available dataset groups
  $0 --list-groups

${BLUE}═══════════════════════════════════════════════════════════════════${NC}
EOF
}

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

POLICIES=""
DATASET_GROUP="${DATASET_GROUP:-}"
SINGLE_DATASET="${SINGLE_DATASET:-}"
COMPUTER="${COMPUTER:-$(hostname)}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
STEPS="${STEPS:-}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)
            POLICIES="$2"
            shift 2
            ;;
        --dataset-group)
            DATASET_GROUP="$2"
            shift 2
            ;;
        --dataset)
            SINGLE_DATASET="$2"
            shift 2
            ;;
        --computer)
            COMPUTER="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --list-groups)
            python3 "$DATASET_RESOLVER" --list-groups
            exit 0
            ;;
        --list-policies)
            echo -e "${GREEN}Available Policies:${NC}"
            echo "  xvla       - X-VLA (Vision-Language-Action model)"
            echo "  pi05       - π₀.₅ (Pi0.5 - latest Physical Intelligence model)"
            echo "  pi0        - π₀ (Pi0 - Physical Intelligence base model)"
            echo "  groot      - NVIDIA GR00T N1.5 (humanoid foundation model)"
            echo "  act        - ACT (Action Chunking with Transformers)"
            echo "  wall_oss   - Wall-OSS (Embodied Foundation Model)"
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# ============================================================================
# VALIDATION
# ============================================================================

# Validate policies
if [ -z "$POLICIES" ]; then
    echo -e "${RED}Error: --policy is required${NC}"
    usage
    exit 1
fi

# Split comma-separated policies into array
IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"

# Validate each policy
VALID_POLICIES=("xvla" "pi05" "pi0" "groot" "act" "wall_oss")
for POLICY in "${POLICY_ARRAY[@]}"; do
    # Trim whitespace
    POLICY=$(echo "$POLICY" | xargs)
    if [[ ! " ${VALID_POLICIES[@]} " =~ " ${POLICY} " ]]; then
        echo -e "${RED}Error: Invalid policy '$POLICY'${NC}"
        echo -e "Valid policies: ${VALID_POLICIES[*]}"
        exit 1
    fi
    # Validate training script exists
    TRAIN_SCRIPT="$SCRIPT_DIR/$POLICY/train.sh"
    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo -e "${RED}Error: Training script not found: $TRAIN_SCRIPT${NC}"
        exit 1
    fi
done

# Validate dataset source (must have either dataset-group or dataset)
if [ -z "$DATASET_GROUP" ] && [ -z "$SINGLE_DATASET" ]; then
    echo -e "${RED}Error: Either --dataset-group or --dataset is required${NC}"
    usage
    exit 1
fi

if [ -n "$DATASET_GROUP" ] && [ -n "$SINGLE_DATASET" ]; then
    echo -e "${RED}Error: Cannot specify both --dataset-group and --dataset${NC}"
    usage
    exit 1
fi

# Validate dataset group if provided
if [ -n "$DATASET_GROUP" ]; then
    if ! python3 "$DATASET_RESOLVER" "$DATASET_GROUP" --format bash > /dev/null 2>&1; then
        echo -e "${RED}Error: Invalid dataset group '$DATASET_GROUP'${NC}"
        echo ""
        python3 "$DATASET_RESOLVER" --list-groups
        exit 1
    fi
fi

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Training Configuration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Policies:${NC}      ${POLICIES} (${#POLICY_ARRAY[@]} model(s))"
if [ -n "$DATASET_GROUP" ]; then
    echo -e "${YELLOW}Dataset Group:${NC} $DATASET_GROUP"
else
    echo -e "${YELLOW}Single Dataset:${NC} $SINGLE_DATASET"
fi
echo -e "${YELLOW}Computer:${NC}      $COMPUTER"
[ -n "$BATCH_SIZE" ] && echo -e "${YELLOW}Batch Size:${NC}    $BATCH_SIZE (override)"
[ -n "$NUM_WORKERS" ] && echo -e "${YELLOW}Num Workers:${NC}   $NUM_WORKERS (override)"
[ -n "$STEPS" ] && echo -e "${YELLOW}Steps:${NC}         $STEPS (override)"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Get dataset info
if [ -n "$DATASET_GROUP" ]; then
    DATASET_COUNT=$(python3 "$DATASET_RESOLVER" "$DATASET_GROUP" --format list | wc -l)
    echo -e "${GREEN}Dataset group '$DATASET_GROUP' contains $DATASET_COUNT dataset(s)${NC}"
else
    echo -e "${GREEN}Training on single dataset: $SINGLE_DATASET${NC}"
fi
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN - Would execute sequentially:${NC}"
    for POLICY in "${POLICY_ARRAY[@]}"; do
        POLICY=$(echo "$POLICY" | xargs)
        TRAIN_SCRIPT="$SCRIPT_DIR/$POLICY/train.sh"
        echo ""
        echo -e "${BLUE}[$POLICY]${NC}"
        echo "COMPUTER=$COMPUTER \\"
        if [ -n "$DATASET_GROUP" ]; then
            echo "DATASET_GROUP=$DATASET_GROUP \\"
        else
            echo "SINGLE_DATASET=$SINGLE_DATASET \\"
        fi
        [ -n "$BATCH_SIZE" ] && echo "BATCH_SIZE=$BATCH_SIZE \\"
        [ -n "$NUM_WORKERS" ] && echo "NUM_WORKERS=$NUM_WORKERS \\"
        [ -n "$STEPS" ] && echo "STEPS=$STEPS \\"
        echo "$TRAIN_SCRIPT"
    done
    exit 0
fi

# ============================================================================
# EXPORT ENVIRONMENT VARIABLES AND RUN TRAINING SEQUENTIALLY
# ============================================================================

echo -e "${GREEN}Starting sequential training for ${#POLICY_ARRAY[@]} policy(ies)...${NC}"
echo ""

export COMPUTER="$COMPUTER"
if [ -n "$DATASET_GROUP" ]; then
    export DATASET_GROUP="$DATASET_GROUP"
else
    export SINGLE_DATASET="$SINGLE_DATASET"
fi
[ -n "$BATCH_SIZE" ] && export BATCH_SIZE="$BATCH_SIZE"
[ -n "$NUM_WORKERS" ] && export NUM_WORKERS="$NUM_WORKERS"
[ -n "$STEPS" ] && export STEPS="$STEPS"

# Train each policy sequentially
POLICY_NUM=1
TOTAL_POLICIES=${#POLICY_ARRAY[@]}
for POLICY in "${POLICY_ARRAY[@]}"; do
    POLICY=$(echo "$POLICY" | xargs)
    TRAIN_SCRIPT="$SCRIPT_DIR/$POLICY/train.sh"
    
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Training Policy $POLICY_NUM/$TOTAL_POLICIES: $POLICY${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Execute training script (it will run in background via nohup)
    "$TRAIN_SCRIPT"
    
    # Wait a moment for the training to start and log file to be created
    sleep 2
    
    # If this is not the last policy, inform user
    if [ $POLICY_NUM -lt $TOTAL_POLICIES ]; then
        echo ""
        echo -e "${YELLOW}Policy $POLICY launched in background. Moving to next policy...${NC}"
        echo ""
    fi
    
    ((POLICY_NUM++))
done

echo ""
echo -e "${GREEN}All ${TOTAL_POLICIES} training job(s) launched!${NC}"
echo -e "Check the training logs in outputs/logs/ for progress."
