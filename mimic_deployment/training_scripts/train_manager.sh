#!/bin/bash
# Master Training Orchestrator for Mimic Robot
# Manages multi-policy training across multiple computers with dataset grouping
# Supports sequential training on multiple dataset groups with auto-upload to Hugging Face

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
${GREEN}Mimic Robot Training Manager (Multi-Group + Auto-Upload)${NC}
${BLUE}═══════════════════════════════════════════════════════════════════${NC}

Usage: $0 [OPTIONS]

${YELLOW}Required Options:${NC}
  --policy POLICY[,POLICY2,...]  Policy type(s) to train (comma-separated for multiple)
                                  Choices: xvla, pi05, pi0, groot, act, wall_oss, smolvla,pi0fast
                                  Example: --policy xvla,pi05 (trains both sequentially)
  
  --dataset-group GROUP[,GROUP2,...]  Dataset group(s) to train on (comma-separated for multiple)
                                      Trains sequentially on each group
                                      Run with --list-groups to see available groups
  
  --dataset REPO_ID               Single dataset repository ID (alternative to --dataset-group)
                                  Example: --dataset Mimic-Robotics/dataset_name

${YELLOW}Optional:${NC}
  --computer NAME          Computer name (default: auto-detect from hostname)
  --batch-size SIZE        Override default batch size
  --num-workers N          Override default number of workers
  --steps N                Number of training steps
  --push-to-hub           Auto-upload model to Hugging Face Hub after training (default: true)
  --list-groups            List all available dataset groups and exit
  --list-policies          List all available policies and exit
  --dry-run                Show configuration without starting training
  -h, --help               Show this help message

${YELLOW}Environment Variables:${NC}
  COMPUTER                 Same as --computer flag
  DATASET_GROUP            Same as --dataset-group flag (comma-separated for multiple)
  BATCH_SIZE               Same as --batch-size flag
  NUM_WORKERS              Same as --num-workers flag
  STEPS                    Same as --steps flag
  PUSH_TO_HUB              Same as --push-to-hub flag (true/false)

${YELLOW}Examples:${NC}
  # Train xVLA on all datasets
  $0 --policy xvla --dataset-group mimic_displacement

  # Train multiple policies sequentially on same dataset group
  $0 --policy xvla,pi05 --dataset-group high_quality

  # Train on multiple groups sequentially with auto-upload
  $0 --policy xvla --dataset-group high_quality,most_recent --push-to-hub --wait-for-completion

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
DATASET_GROUPS="${DATASET_GROUP:-}"
SINGLE_DATASET="${SINGLE_DATASET:-}"
COMPUTER="${COMPUTER:-$(hostname)}"
if [[ "$COMPUTER" == speed-* ]] || [[ "$COMPUTER" == speed*.* ]]; then
    COMPUTER="speed"
fi
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
STEPS="${STEPS:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
DRY_RUN=false
RUN_BACKGROUND=true # Default behavio
TIMESTAMP=$(date +"%Y%m%d_%H%M") # Format: 20260210_1521

# SLURM defaults for Speed cluster
SLURM_PARTITION="${SLURM_PARTITION:-pg}"
SLURM_GPUS="${SLURM_GPUS:-4}"
SLURM_CPUS="${SLURM_CPUS:-32}"
SLURM_MEM="${SLURM_MEM:-128G}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)
            POLICIES="$2"
            shift 2
            ;;
        --dataset-group)
            DATASET_GROUPS="$2"
            shift 2
            ;;
        --dataset)
            SINGLE_DATASET="$2"
            shift 2
            ;;
        --computer)
            COMPUTER="$2"
            if [[ "$COMPUTER" == speed-* ]] || [[ "$COMPUTER" == speed*.* ]]; then
                COMPUTER="speed"
            fi
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
        --action-steps)
            ACTION_STEPS="$2"
            shift 2
            ;;       
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --partition)
            SLURM_PARTITION="$2"
            shift 2
            ;;
        --gpus)
            SLURM_GPUS="$2"
            shift 2
            ;;
        --time-limit)
            SLURM_TIME="$2"
            shift 2
            ;;
        --slurm-mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        --push-to-hub)
            PUSH_TO_HUB=true
            shift
            ;;
        --noback)
            RUN_BACKGROUND=false
            shift
            ;;
        --list-groups)
            python3 "$DATASET_RESOLVER" --list-groups
            exit 0
            ;;
        --list-policies)
            echo -e "${GREEN}Available Policies:${NC}"
            echo "  xvla       - X-VLA (Vision-Language-Action model)"
            echo "  pi05       - π₀.₅ (Pi0.5 - latest Physical Intelligence model)"
            echo "  pi0fast   - π₀.₅ Fast (Pi0 - faster training version)"
            echo "  pi0        - π₀ (Pi0 - Physical Intelligence base model)"
            echo "  groot      - NVIDIA GR00T N1.5 (humanoid foundation model)"
            echo "  act        - ACT (Action Chunking with Transformers)"
            echo "  smolvla    - SmolVLA (Efficient Vision-Language-Action model)"
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

# Split comma-separated policies and groups into arrays
IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"
IFS=',' read -ra GROUP_ARRAY <<< "$DATASET_GROUPS"

# Validate each policy
VALID_POLICIES=("xvla" "pi05" "pi0" "groot" "act" "wall_oss" "smolvla" "pi0fast")
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
if [ -z "$DATASET_GROUPS" ] && [ -z "$SINGLE_DATASET" ]; then
    echo -e "${RED}Error: Either --dataset-group or --dataset is required${NC}"
    usage
    exit 1
fi

if [ -n "$DATASET_GROUPS" ] && [ -n "$SINGLE_DATASET" ]; then
    echo -e "${RED}Error: Cannot specify both --dataset-group and --dataset${NC}"
    usage
    exit 1
fi

# Validate each dataset group if provided
if [ -n "$DATASET_GROUPS" ]; then
    for GROUP in "${GROUP_ARRAY[@]}"; do
        GROUP=$(echo "$GROUP" | xargs)
        if ! python3 "$DATASET_RESOLVER" "$GROUP" --format bash > /dev/null 2>&1; then
            echo -e "${RED}Error: Invalid dataset group '$GROUP'${NC}"
            echo ""
            python3 "$DATASET_RESOLVER" --list-groups
            exit 1
        fi
    done
fi

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Training Configuration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Policies:${NC}           ${POLICIES} (${#POLICY_ARRAY[@]} model(s))"
if [ -n "$DATASET_GROUPS" ]; then
    echo -e "${YELLOW}Dataset Groups:${NC}     $DATASET_GROUPS (${#GROUP_ARRAY[@]} group(s))"
else
    echo -e "${YELLOW}Single Dataset:${NC}     $SINGLE_DATASET"
fi
echo -e "${YELLOW}Computer:${NC}           $COMPUTER"
[ -n "$BATCH_SIZE" ] && echo -e "${YELLOW}Batch Size:${NC}         $BATCH_SIZE (override)"
[ -n "$NUM_WORKERS" ] && echo -e "${YELLOW}Num Workers:${NC}        $NUM_WORKERS (override)"
[ -n "$STEPS" ] && echo -e "${YELLOW}Steps:${NC}              $STEPS (override)"
echo -e "${YELLOW}Push to Hub:${NC}        $PUSH_TO_HUB"
echo -e "${YELLOW}Run Background:${NC}     $RUN_BACKGROUND"
echo -e "${YELLOW}Timestamp:${NC}          $TIMESTAMP"
if [ "$COMPUTER" = "speed" ]; then
    echo -e "${YELLOW}SLURM Partition:${NC}    $SLURM_PARTITION"
    echo -e "${YELLOW}SLURM GPUs:${NC}         $SLURM_GPUS"
    echo -e "${YELLOW}SLURM CPUs:${NC}         $SLURM_CPUS"
    echo -e "${YELLOW}SLURM Memory:${NC}       $SLURM_MEM"
    echo -e "${YELLOW}SLURM Time:${NC}         $SLURM_TIME"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Get dataset info
if [ -n "$DATASET_GROUPS" ]; then
    TOTAL_DATASETS=0
    for GROUP in "${GROUP_ARRAY[@]}"; do
        GROUP=$(echo "$GROUP" | xargs)
        GROUP_COUNT=$(python3 "$DATASET_RESOLVER" "$GROUP" --format list | wc -l)
        echo -e "${GREEN}Dataset group '$GROUP' contains $GROUP_COUNT dataset(s)${NC}"
        TOTAL_DATASETS=$((TOTAL_DATASETS + GROUP_COUNT))
    done
    echo -e "${GREEN}Total datasets across all groups: $TOTAL_DATASETS${NC}"
else
    echo -e "${GREEN}Training on single dataset: $SINGLE_DATASET${NC}"
fi
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN - Training sequence:${NC}"
    echo ""
    if [ -n "$DATASET_GROUPS" ]; then
        GROUP_NUM=1
        for GROUP in "${GROUP_ARRAY[@]}"; do
            GROUP=$(echo "$GROUP" | xargs)
            echo -e "${BLUE}Dataset Group $GROUP_NUM/${#GROUP_ARRAY[@]}: $GROUP${NC}"
            POLICY_NUM=1
            for POLICY in "${POLICY_ARRAY[@]}"; do
                POLICY=$(echo "$POLICY" | xargs)
                echo -e "  ${GREEN}├─ Policy $POLICY_NUM/${#POLICY_ARRAY[@]}: $POLICY${NC}"
                ((POLICY_NUM++))
            done
            ((GROUP_NUM++))
            echo ""
        done
    else
        POLICY_NUM=1
        for POLICY in "${POLICY_ARRAY[@]}"; do
            POLICY=$(echo "$POLICY" | xargs)
            echo -e "${GREEN}Policy $POLICY_NUM/${#POLICY_ARRAY[@]}: $POLICY${NC}"
            echo -e "  Dataset: $SINGLE_DATASET"
            ((POLICY_NUM++))
        done
    fi
    exit 0
fi

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

echo -e "${GREEN}Starting training...${NC}"
echo ""

PREV_SLURM_JOB_ID=""
export COMPUTER="$COMPUTER"
[ -n "$BATCH_SIZE" ] && export BATCH_SIZE="$BATCH_SIZE"
[ -n "$NUM_WORKERS" ] && export NUM_WORKERS="$NUM_WORKERS"
[ -n "$STEPS" ] && export STEPS="$STEPS"
[ -n "$ACTION_STEPS" ] && export ACTION_STEPS="$ACTION_STEPS"
[ -n "$CHUNK_SIZE" ] && export CHUNK_SIZE="$CHUNK_SIZE"


# Determine the dataset loop
if [ -n "$DATASET_GROUPS" ]; then
    OUTER_ARRAY=("${GROUP_ARRAY[@]}")
    MODE="GROUP"
else
    OUTER_ARRAY=("$SINGLE_DATASET")
    MODE="SINGLE"
fi

TOTAL_OUTER=${#OUTER_ARRAY[@]}
OUTER_NUM=1

for ITEM in "${OUTER_ARRAY[@]}"; do
    ITEM=$(echo "$ITEM" | xargs)
    
    if [ "$MODE" = "GROUP" ]; then
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}Training on Dataset Group $OUTER_NUM/$TOTAL_OUTER: $ITEM${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
        export DATASET_GROUP="$ITEM"
        unset SINGLE_DATASET
        DATASET_NAME_CLEAN=$(echo "$ITEM" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    else
        export SINGLE_DATASET="$ITEM"
        unset DATASET_GROUP
        DATASET_NAME_CLEAN=$(echo "$ITEM" | sed 's|.*/||')
    fi

    POLICY_NUM=1
    TOTAL_POLICIES=${#POLICY_ARRAY[@]}
    
    for POLICY in "${POLICY_ARRAY[@]}"; do
        POLICY=$(echo "$POLICY" | xargs)
        TRAIN_SCRIPT="$SCRIPT_DIR/$POLICY/train.sh"
        
        export JOB_NAME="${POLICY}_${COMPUTER}_${DATASET_NAME_CLEAN}_${TIMESTAMP}"
        
        echo -e "${YELLOW}Policy $POLICY_NUM/$TOTAL_POLICIES: $POLICY (Job: $JOB_NAME)${NC}"

        if [ "$COMPUTER" = "speed" ]; then
            # === SLURM SBATCH SUBMISSION ===
            SBATCH_SCRIPT="$REPO_ROOT/outputs/logs/${JOB_NAME}.sbatch"
            cat > "$SBATCH_SCRIPT" << SBATCH_EOF
#!/encs/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${REPO_ROOT}/outputs/logs/${JOB_NAME}_%j.out
#SBATCH --error=${REPO_ROOT}/outputs/logs/${JOB_NAME}_%j.err
#SBATCH --mail-type=ALL

mkdir -p /speed-scratch/\$USER/tmp
export TMPDIR=/speed-scratch/\$USER/tmp
export TMP=\$TMPDIR

export COMPUTER="${COMPUTER}"
export JOB_NAME="${JOB_NAME}"
export BATCH_SIZE="${BATCH_SIZE}"
export NUM_WORKERS="${NUM_WORKERS}"
export STEPS="${STEPS}"
export ACTION_STEPS="${ACTION_STEPS:-}"
export CHUNK_SIZE="${CHUNK_SIZE:-}"
export PUSH_TO_HUB="${PUSH_TO_HUB}"
export DATASET_GROUP="${DATASET_GROUP:-}"
export SINGLE_DATASET="${SINGLE_DATASET:-}"

${TRAIN_SCRIPT} --no-daemon
SBATCH_EOF
            chmod +x "$SBATCH_SCRIPT"

            SBATCH_ARGS=""
            [ -n "${PREV_SLURM_JOB_ID:-}" ] && SBATCH_ARGS="--depend=afterany:$PREV_SLURM_JOB_ID"
            SLURM_OUTPUT=$(sbatch $SBATCH_ARGS "$SBATCH_SCRIPT")
            PREV_SLURM_JOB_ID=$(echo "$SLURM_OUTPUT" | grep -oP '\d+$')
            echo -e "${GREEN}Submitted SLURM job: $SLURM_OUTPUT${NC}"
            echo -e "${GREEN}Script: $SBATCH_SCRIPT${NC}"

        elif [ "$RUN_BACKGROUND" = true ]; then
            # Execute training script in background
            "$TRAIN_SCRIPT"
            
            PID_FILE="$REPO_ROOT/outputs/logs/${JOB_NAME}.pid"
            sleep 2
            if [ -f "$PID_FILE" ]; then
                TRAIN_PID=$(cat "$PID_FILE")
                echo -e "${YELLOW}Waiting for background training (PID: $TRAIN_PID)...${NC}"
                wait "$TRAIN_PID" 2>/dev/null || true
                rm -f "$PID_FILE"
            else
                echo -e "${RED}Warning: PID file not found for $JOB_NAME${NC}"
            fi
        else
            # Execute training script in foreground
            echo -e "${BLUE}Running in foreground...${NC}"
            "$TRAIN_SCRIPT" --no-daemon
        fi
        
        echo -e "${GREEN}Completed: $POLICY${NC}"
        echo ""
        ((POLICY_NUM++))
    done
    
    ((OUTER_NUM++))
done

echo -e "${GREEN}All training sessions completed!${NC}"
if [ "$COMPUTER" = "speed" ] && [ -n "$PREV_SLURM_JOB_ID" ]; then
    echo -e "${YELLOW}Monitor jobs: squeue -u \$USER${NC}"
fi