#!/bin/bash
# SPEED-only Training Manager (SBATCH only)
# Simple, robust orchestration with readable logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET_RESOLVER="$SCRIPT_DIR/dataset_groups.py"
OUTPUT_BASE="${OUTPUT_BASE:-/speed-scratch/$USER/mimic-lerobot-outputs}"
LOG_DIR="$OUTPUT_BASE/logs"
TRAIN_DIR="$OUTPUT_BASE/train"
TIMESTAMP="$(date +"%d_%b_%H%M%S" | tr '[:upper:]' '[:lower:]')"
SUMMARY_LOG="$LOG_DIR/train_manager_speed_${TIMESTAMP}.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
  cat << EOF
${BLUE}═══════════════════════════════════════════════════════════════════${NC}
${GREEN}SPEED Training Manager (SBATCH only)${NC}
${BLUE}═══════════════════════════════════════════════════════════════════${NC}

Usage: $0 [OPTIONS]

${YELLOW}Required:${NC}
  --policy POLICY[,POLICY2,...]          Policy(s): xvla, pi05, pi0, pi0fast, groot, act, smolvla, wall_oss
  --dataset-group GROUP[,GROUP2,...]     Dataset group(s) from dataset_groups.py

${YELLOW}Core options:${NC}
  --steps N               Training steps (default: 300000)
  --checkpoint-freq N     Checkpoint frequency in steps (default: 50000)
  --batch-size N          Override batch size
  --job-tag TAG           Append custom tag to generated job names
  --policy-mode MODE      Policy mode (default|smoke1k|maxbatch), default: default
  --gpus N                Number of GPUs for SBATCH (default: 1)
  --slurm-mem SIZE        SLURM memory request (default: 256G)

${YELLOW}Utility:${NC}
  --no-follow             Do not stream job logs in this terminal
  --push-to-hub           Force push to hub (default: true)
  --no-push-to-hub        Disable push to hub
  --list-groups           List dataset groups and exit
  --list-policies         List policies and exit
  --dry-run               Show what would be submitted
  -h, --help              Show help

${YELLOW}Examples:${NC}
  # Simple (hassle-free)
  $0 --policy xvla --dataset-group redx_full_vlm

  # More control (GPUs, memory, steps, batch size)
  $0 --policy xvla,pi05 --dataset-group redx_full_vlm,most_recent --gpus 2 --slurm-mem 256G --steps 300000 --batch-size 32

${YELLOW}Notes:${NC}
  - This script always runs on SPEED with SBATCH only.
  - Uses one plain .log file per job (no separate .err file).
  - Default SLURM partition/time are set in-script for simplicity.
EOF
}

POLICIES=""
DATASET_GROUPS="${DATASET_GROUP:-}"
STEPS="${STEPS:-}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-${SAVE_FREQ:-}}"
BATCH_SIZE="${BATCH_SIZE:-}"
JOB_TAG="${JOB_TAG:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
FOLLOW_LOGS=true
DRY_RUN=false
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot}"
XVLA_CONDA_ENV_NAME="${XVLA_CONDA_ENV_NAME:-/speed-scratch/$USER/conda/lerobot-xvla}"
PI05_CONDA_ENV_NAME="${PI05_CONDA_ENV_NAME:-/speed-scratch/$USER/conda/lerobot-pi}"
POLICY_MODE="${POLICY_MODE:-default}"

# SPEED defaults (kept simple; edit here if cluster policy changes)
SLURM_PARTITION="pt"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_GRES="${SLURM_GRES-gpu:nvidia_a100_7g.80gb:1}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-256G}"
SLURM_TIME="${SLURM_TIME:-3-00:00:00}"
HF_PUSH_CHECKPOINTS="${HF_PUSH_CHECKPOINTS:-true}"
HF_CHECKPOINT_SYNC_INTERVAL="${HF_CHECKPOINT_SYNC_INTERVAL:-180}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-false}"

STEPS="${STEPS:-300000}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-50000}"

ensure_conda_active() {
  if [ "${SKIP_LOCAL_CONDA_CHECK:-false}" = "true" ]; then
    return 0
  fi

  if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV_NAME" ] || [[ "${CONDA_PREFIX:-}" == */"$CONDA_ENV_NAME" ]]; then
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
  fi

  for conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "/speed-scratch/$USER/conda/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  do
    if [ -f "$conda_sh" ]; then
      source "$conda_sh"
      break
    fi
  done

  conda activate "$CONDA_ENV_NAME" >/dev/null 2>&1 || {
    echo -e "${RED}Error: failed to activate conda env '$CONDA_ENV_NAME'.${NC}"
    return 1
  }

  return 0
}

ensure_conda_active

PYTHON_BIN="$(command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [ -z "$PYTHON_BIN" ]; then
  echo -e "${RED}Error: python not found after conda activation.${NC}"
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      POLICIES="$2"
      shift 2
      ;;
    --dataset-group)
      DATASET_GROUPS="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --checkpoint-freq)
      CHECKPOINT_FREQ="$2"
      shift 2
      ;;
    --save-freq)
      CHECKPOINT_FREQ="$2"
      echo -e "${YELLOW}Warning:${NC} --save-freq is deprecated; use --checkpoint-freq."
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --job-tag)
      JOB_TAG="$2"
      shift 2
      ;;
    --policy-mode)
      POLICY_MODE="$2"
      shift 2
      ;;
    --gpus)
      SLURM_GPUS="$2"
      shift 2
      ;;
    --slurm-mem)
      SLURM_MEM="$2"
      shift 2
      ;;
    --no-follow)
      FOLLOW_LOGS=false
      shift
      ;;
    --push-to-hub)
      PUSH_TO_HUB=true
      shift
      ;;
    --no-push-to-hub)
      PUSH_TO_HUB=false
      shift
      ;;
    --list-groups)
      "$PYTHON_BIN" "$DATASET_RESOLVER" --list-groups
      exit 0
      ;;
    --list-policies)
      echo -e "${GREEN}Available Policies:${NC}"
      echo "  xvla"
      echo "  pi05"
      echo "  pi0"
      echo "  pi0fast"
      echo "  groot"
      echo "  act"
      echo "  smolvla"
      echo "  wall_oss"
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

if [ "$PUSH_TO_HUB" != "true" ]; then
  echo -e "${YELLOW}Note:${NC} SPEED manager enforces push_to_hub=true; overriding requested value."
  PUSH_TO_HUB=true
fi

if [ -z "$POLICIES" ]; then
  echo -e "${RED}Error: --policy is required${NC}"
  usage
  exit 1
fi

if [[ "$POLICY_MODE" != "default" && "$POLICY_MODE" != "smoke1k" && "$POLICY_MODE" != "maxbatch" ]]; then
  echo -e "${RED}Error: --policy-mode must be one of: default, smoke1k, maxbatch${NC}"
  exit 1
fi

if [[ "$POLICY_MODE" == "smoke1k" ]]; then
  STEPS="1000"
  CHECKPOINT_FREQ="1000"
  WANDB_DISABLE_ARTIFACT="true"
fi

if [[ "$POLICY_MODE" == "maxbatch" ]]; then
  WANDB_DISABLE_ARTIFACT="true"
fi

if ! [[ "$SLURM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo -e "${RED}Error: --gpus must be a positive integer (>=1).${NC}"
  exit 1
fi

if [ -z "$DATASET_GROUPS" ]; then
  echo -e "${RED}Error: --dataset-group is required${NC}"
  usage
  exit 1
fi

IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"
IFS=',' read -ra GROUP_ARRAY <<< "$DATASET_GROUPS"

VALID_POLICIES=("xvla" "pi05" "pi0" "groot" "act" "wall_oss" "smolvla" "pi0fast")
for POLICY in "${POLICY_ARRAY[@]}"; do
  POLICY="$(echo "$POLICY" | xargs)"
  if [[ ! " ${VALID_POLICIES[*]} " =~ " ${POLICY} " ]]; then
    echo -e "${RED}Error: Invalid policy '$POLICY'${NC}"
    echo "Valid policies: ${VALID_POLICIES[*]}"
    exit 1
  fi
  TRAIN_SCRIPT="$SCRIPT_DIR/$POLICY/train.sh"
  if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo -e "${RED}Error: Missing training script: $TRAIN_SCRIPT${NC}"
    exit 1
  fi
done

for GROUP in "${GROUP_ARRAY[@]}"; do
  GROUP="$(echo "$GROUP" | xargs)"
  if ! "$PYTHON_BIN" "$DATASET_RESOLVER" "$GROUP" --format bash > /dev/null 2>&1; then
    echo -e "${RED}Error: Invalid dataset group '$GROUP'${NC}"
    "$PYTHON_BIN" "$DATASET_RESOLVER" --list-groups
    exit 1
  fi
done

mkdir -p "$LOG_DIR" "$TRAIN_DIR"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}SPEED SBATCH Configuration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Policies:${NC}       $POLICIES (${#POLICY_ARRAY[@]} model(s))"
echo -e "${YELLOW}Dataset Groups:${NC} $DATASET_GROUPS (${#GROUP_ARRAY[@]} group(s))"
echo -e "${YELLOW}Steps:${NC}          $STEPS"
echo -e "${YELLOW}Checkpoint Freq:${NC} $CHECKPOINT_FREQ"
echo -e "${YELLOW}Batch Size:${NC}     ${BATCH_SIZE:-<policy default>}"
echo -e "${YELLOW}Job Tag:${NC}        ${JOB_TAG:-<none>}"
echo -e "${YELLOW}Policy Mode:${NC}    $POLICY_MODE"
echo -e "${YELLOW}GPUs:${NC}           $SLURM_GPUS"
echo -e "${YELLOW}Constraint:${NC}     ${SLURM_CONSTRAINT:-<none>}"
echo -e "${YELLOW}GRES Override:${NC}  ${SLURM_GRES:-<none>}"
echo -e "${YELLOW}Output Base:${NC}    $OUTPUT_BASE"
echo -e "${YELLOW}SLURM Mem:${NC}      $SLURM_MEM"
echo -e "${YELLOW}Conda (default):${NC} $CONDA_ENV_NAME"
echo -e "${YELLOW}Conda (xvla):${NC}   $XVLA_CONDA_ENV_NAME"
echo -e "${YELLOW}Conda (pi05):${NC}   $PI05_CONDA_ENV_NAME"
echo -e "${YELLOW}Push to Hub:${NC}    $PUSH_TO_HUB"
echo -e "${YELLOW}Push Checkpoints:${NC} $HF_PUSH_CHECKPOINTS (every ${HF_CHECKPOINT_SYNC_INTERVAL}s)"
echo -e "${YELLOW}W&B Artifacts:${NC}  ${WANDB_DISABLE_ARTIFACT} (disable_artifact)"
echo -e "${YELLOW}Follow Logs:${NC}    $FOLLOW_LOGS"
echo -e "${YELLOW}Summary Log:${NC}    $SUMMARY_LOG"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

touch "$SUMMARY_LOG"
{
  echo "[$(date '+%F %T')] SPEED training manager started"
  echo "Policies=$POLICIES"
  echo "DatasetGroups=$DATASET_GROUPS"
  echo "Steps=$STEPS"
  echo "CheckpointFreq=$CHECKPOINT_FREQ"
  echo "BatchSize=${BATCH_SIZE:-<policy default>}"
  echo "JobTag=${JOB_TAG:-<none>}"
  echo "PolicyMode=$POLICY_MODE"
  echo "GPUs=$SLURM_GPUS"
  echo "Constraint=${SLURM_CONSTRAINT:-<none>}"
  echo "GresOverride=${SLURM_GRES:-<none>}"
  echo "OutputBase=$OUTPUT_BASE"
  echo "SlurmMem=$SLURM_MEM"
  echo "CondaDefault=$CONDA_ENV_NAME"
  echo "CondaXvla=$XVLA_CONDA_ENV_NAME"
  echo "CondaPi05=$PI05_CONDA_ENV_NAME"
  echo "PushToHub=$PUSH_TO_HUB"
  echo "PushCheckpoints=$HF_PUSH_CHECKPOINTS"
  echo "CheckpointSyncInterval=$HF_CHECKPOINT_SYNC_INTERVAL"
  echo "WandbDisableArtifact=$WANDB_DISABLE_ARTIFACT"
} >> "$SUMMARY_LOG"

if [ "$DRY_RUN" = true ]; then
  echo -e "${YELLOW}DRY RUN - Submission order:${NC}"
  for GROUP in "${GROUP_ARRAY[@]}"; do
    GROUP="$(echo "$GROUP" | xargs)"
    echo -e "${BLUE}Group:${NC} $GROUP"
    "$PYTHON_BIN" "$DATASET_RESOLVER" "$GROUP" --format list | sed 's/^/  - /'
    for POLICY in "${POLICY_ARRAY[@]}"; do
      POLICY="$(echo "$POLICY" | xargs)"
      echo "  -> $POLICY"
    done
  done
  exit 0
fi

extract_wandb_link() {
  local log_file="$1"
  grep -Eo 'https?://[^ ]*wandb.ai[^ ]*' "$log_file" 2>/dev/null | tail -n 1 || true
}

follow_job_log() {
  local job_id="$1"
  local log_file="$2"

  touch "$log_file"
  echo -e "${GREEN}Streaming job ${job_id} log:${NC} $log_file"

  tail -n 50 -F "$log_file" &
  local tail_pid=$!

  while true; do
    if ! squeue -h -j "$job_id" | grep -q "$job_id"; then
      break
    fi
    sleep 10
  done

  sleep 2
  kill "$tail_pid" 2>/dev/null || true
  wait "$tail_pid" 2>/dev/null || true

  local wandb_link
  wandb_link="$(extract_wandb_link "$log_file")"
  if [ -n "$wandb_link" ]; then
    echo -e "${GREEN}W&B:${NC} $wandb_link"
  else
    echo -e "${YELLOW}W&B link not found yet in log.${NC}"
  fi

  if grep -qiE 'torch\.OutOfMemoryError|CUDA out of memory|out of memory' "$log_file"; then
    echo -e "${RED}Detected OOM in log.${NC} Lower --batch-size (try 4 or 2 for xvla)."
  fi

  if grep -qi 'Traceback (most recent call last)' "$log_file"; then
    echo -e "${YELLOW}Detected Python traceback in log.${NC} tail -n 120 $log_file"
  fi
}

PREV_JOB_ID=""
GROUP_NUM=1

SBATCH_GPU_LINE="#SBATCH --gpus=${SLURM_GPUS}"
if [ -n "$SLURM_GRES" ]; then
  SBATCH_GPU_LINE="#SBATCH --gres=${SLURM_GRES}"
fi

SBATCH_CONSTRAINT_LINE=""
if [ -n "$SLURM_CONSTRAINT" ]; then
  SBATCH_CONSTRAINT_LINE="#SBATCH --constraint=${SLURM_CONSTRAINT}"
fi

for GROUP in "${GROUP_ARRAY[@]}"; do
  GROUP="$(echo "$GROUP" | xargs)"
  DATASET_NAME_CLEAN="$(echo "$GROUP" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr ',' '_')"

  echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
  echo -e "${GREEN}Dataset Group $GROUP_NUM/${#GROUP_ARRAY[@]}: $GROUP${NC}"
  echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

  echo -e "${YELLOW}Datasets in group:${NC}"
  "$PYTHON_BIN" "$DATASET_RESOLVER" "$GROUP" --format list | sed 's/^/  - /'

  POLICY_NUM=1
  for POLICY in "${POLICY_ARRAY[@]}"; do
    POLICY="$(echo "$POLICY" | xargs)"
    POLICY_CONDA_ENV_NAME="$CONDA_ENV_NAME"
    if [ "$POLICY" = "xvla" ]; then
      POLICY_CONDA_ENV_NAME="$XVLA_CONDA_ENV_NAME"
    fi
    if [ "$POLICY" = "pi05" ]; then
      POLICY_CONDA_ENV_NAME="$PI05_CONDA_ENV_NAME"
    fi

    EFFECTIVE_BATCH_SIZE="$BATCH_SIZE"
    if [ -z "$EFFECTIVE_BATCH_SIZE" ] && [ "$POLICY" = "xvla" ]; then
      EFFECTIVE_BATCH_SIZE="4"
      echo -e "${YELLOW}Auto batch-size for xvla:${NC} 4"
    fi

    TRAIN_SCRIPT="$SCRIPT_DIR/$POLICY/train.sh"
    BATCH_TAG="${EFFECTIVE_BATCH_SIZE:-auto}"
    JOB_TAG_CLEAN="$(echo "$JOB_TAG" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd 'a-z0-9_-')"
    if [ -n "$JOB_TAG_CLEAN" ]; then
      JOB_NAME="${POLICY}_speed_${DATASET_NAME_CLEAN}_${JOB_TAG_CLEAN}_b${BATCH_TAG}_${TIMESTAMP}"
    else
      JOB_NAME="${POLICY}_speed_${DATASET_NAME_CLEAN}_b${BATCH_TAG}_${TIMESTAMP}"
    fi
    SLURM_SCRIPT="$LOG_DIR/${JOB_NAME}.slurm.sh"
    JOB_LOG="$LOG_DIR/${JOB_NAME}.log"

    echo -e "${YELLOW}Submitting policy $POLICY_NUM/${#POLICY_ARRAY[@]}: $POLICY${NC}"

    cat > "$SLURM_SCRIPT" << SBATCH_EOF
#!/encs/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${SLURM_PARTITION}
${SBATCH_CONSTRAINT_LINE}
${SBATCH_GPU_LINE}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${JOB_LOG}
#SBATCH --error=${JOB_LOG}
#SBATCH --open-mode=append

set -euo pipefail

if [ "\${CONDA_DEFAULT_ENV:-}" != "${POLICY_CONDA_ENV_NAME}" ] && [[ "\${CONDA_PREFIX:-}" != */"${POLICY_CONDA_ENV_NAME}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook 2>/dev/null)" || true
  fi
  if ! command -v conda >/dev/null 2>&1; then
    [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    [ -f "/speed-scratch/\$USER/conda/etc/profile.d/conda.sh" ] && source "/speed-scratch/\$USER/conda/etc/profile.d/conda.sh"
    [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "\$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  if ! conda activate "${POLICY_CONDA_ENV_NAME}" >/dev/null 2>&1; then
    echo "[warn] failed to activate conda env '${POLICY_CONDA_ENV_NAME}'"
    if [ -x "${POLICY_CONDA_ENV_NAME}/bin/python" ]; then
      export PATH="${POLICY_CONDA_ENV_NAME}/bin:\$PATH"
      echo "[info] using python from ${POLICY_CONDA_ENV_NAME}/bin"
    fi
  fi
fi

cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_BASE}/train" "${OUTPUT_BASE}/logs"
export OUTPUT_BASE="${OUTPUT_BASE}"

mkdir -p /speed-scratch/\$USER/tmp
export TMPDIR=/speed-scratch/\$USER/tmp
export TMP=\$TMPDIR
mkdir -p /speed-scratch/\$USER/hf_home
mkdir -p /speed-scratch/\$USER/hf_cache
mkdir -p /speed-scratch/\$USER/wandb
mkdir -p /speed-scratch/\$USER/wandb_cache
mkdir -p /speed-scratch/\$USER/wandb_artifacts
export HF_HOME="\${HF_HOME:-/speed-scratch/\$USER/hf_home}"
export HUGGINGFACE_HUB_CACHE="\${HUGGINGFACE_HUB_CACHE:-/speed-scratch/\$USER/hf_cache}"
export WANDB__SERVICE_WAIT="\${WANDB__SERVICE_WAIT:-300}"
export WANDB_START_METHOD="\${WANDB_START_METHOD:-thread}"
export WANDB_DIR="\${WANDB_DIR:-/speed-scratch/\$USER/wandb}"
export WANDB_CACHE_DIR="\${WANDB_CACHE_DIR:-/speed-scratch/\$USER/wandb_cache}"
export WANDB_ARTIFACT_DIR="\${WANDB_ARTIFACT_DIR:-/speed-scratch/\$USER/wandb_artifacts}"
export WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT}"
export TOKENIZERS_PARALLELISM="\${TOKENIZERS_PARALLELISM:-false}"

export COMPUTER="speed"
export JOB_NAME="${JOB_NAME}"
export DATASET_GROUP="${GROUP}"
export PUSH_TO_HUB="${PUSH_TO_HUB}"
export HF_PUSH_CHECKPOINTS="${HF_PUSH_CHECKPOINTS}"
export HF_CHECKPOINT_SYNC_INTERVAL="${HF_CHECKPOINT_SYNC_INTERVAL}"
export POLICY_MODE="${POLICY_MODE}"

# Probe/maxbatch controls (passed through when set in submission env)
[ -n "${BATCH_CANDIDATES:-}" ] && export BATCH_CANDIDATES="${BATCH_CANDIDATES:-}"
[ -n "${PROBE_STEPS:-}" ] && export PROBE_STEPS="${PROBE_STEPS:-}"
[ -n "${PROBE_SAVE_FREQ:-}" ] && export PROBE_SAVE_FREQ="${PROBE_SAVE_FREQ:-}"
[ -n "${RUN_FINAL_AFTER_PROBE:-}" ] && export RUN_FINAL_AFTER_PROBE="${RUN_FINAL_AFTER_PROBE:-}"
[ -n "${FINAL_STEPS:-}" ] && export FINAL_STEPS="${FINAL_STEPS:-}"
[ -n "${FINAL_SAVE_FREQ:-}" ] && export FINAL_SAVE_FREQ="${FINAL_SAVE_FREQ:-}"
[ -n "${FINAL_BATCH_SIZE:-}" ] && export FINAL_BATCH_SIZE="${FINAL_BATCH_SIZE:-}"

# Pi0.5 controls (passed through when set in submission env)
[ -n "${PI05_FREEZE_VISION_ENCODER:-}" ] && export PI05_FREEZE_VISION_ENCODER="${PI05_FREEZE_VISION_ENCODER:-}"
[ -n "${PI05_TRAIN_EXPERT_ONLY:-}" ] && export PI05_TRAIN_EXPERT_ONLY="${PI05_TRAIN_EXPERT_ONLY:-}"
[ -n "${PI05_GRADIENT_CHECKPOINTING:-}" ] && export PI05_GRADIENT_CHECKPOINTING="${PI05_GRADIENT_CHECKPOINTING:-}"
[ -n "${PI05_COMPILE_MODEL:-}" ] && export PI05_COMPILE_MODEL="${PI05_COMPILE_MODEL:-}"
[ -n "${PI05_COMPILE_MODE:-}" ] && export PI05_COMPILE_MODE="${PI05_COMPILE_MODE:-}"

if [ "${POLICY}" = "xvla" ]; then
  export XVLA_SPEED_MODE="${POLICY_MODE}"
fi
if [ "${POLICY}" = "pi05" ]; then
  export PI05_SPEED_MODE="${POLICY_MODE}"
fi

if [ -n "${EFFECTIVE_BATCH_SIZE}" ]; then export BATCH_SIZE="${EFFECTIVE_BATCH_SIZE}"; fi
if [ -n "${STEPS}" ]; then export STEPS="${STEPS}"; fi
if [ -n "${CHECKPOINT_FREQ}" ]; then export SAVE_FREQ="${CHECKPOINT_FREQ}"; fi

# pass-through auth/env vars if already present
[ -n "\${WANDB_API_KEY:-}" ] && export WANDB_API_KEY
[ -n "\${WANDB_ENTITY:-}" ] && export WANDB_ENTITY
[ -n "\${WANDB_PROJECT:-}" ] && export WANDB_PROJECT
[ -n "\${WANDB_MODE:-}" ] && export WANDB_MODE
[ -n "\${HF_TOKEN:-}" ] && export HF_TOKEN
[ -n "\${HUGGINGFACE_HUB_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN

echo "[info] Job started: \$(date '+%F %T')"
echo "[info] Policy: ${POLICY}"
echo "[info] Policy conda env: ${POLICY_CONDA_ENV_NAME}"
echo "[info] Dataset group: ${GROUP}"
echo "[info] Log file: ${JOB_LOG}"
echo "[info] Conda env: \${CONDA_DEFAULT_ENV:-<none>}"
echo "[info] Python: \$(command -v python || command -v python3 || echo 'missing')"
if ! (python -c "import torch" >/dev/null 2>&1); then
  echo "[error] torch is unavailable in current python environment."
  echo "[error] Set policy conda env correctly (example: /speed-scratch/\$USER/conda/lerobot or /speed-scratch/\$USER/conda/lerobot-pi)."
  exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] GPU memory visible to this job:"
  nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current --format=csv,noheader || true
fi

"${TRAIN_SCRIPT}" --no-daemon

echo "[info] Job finished: \$(date '+%F %T')"
SBATCH_EOF

    chmod +x "$SLURM_SCRIPT"

    SBATCH_ARGS=()
    if [ -n "$PREV_JOB_ID" ]; then
      SBATCH_ARGS+=("--dependency=afterany:${PREV_JOB_ID}")
    fi

    JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SLURM_SCRIPT" | cut -d';' -f1)"
    PREV_JOB_ID="$JOB_ID"

    echo -e "${GREEN}Submitted job:${NC} $JOB_ID"
    echo -e "${GREEN}Policy:${NC}        $POLICY"
    echo -e "${GREEN}Dataset group:${NC} $GROUP"
    echo -e "${GREEN}Log file:${NC}      tail -f $JOB_LOG"
    echo -e "${GREEN}SLURM script:${NC}  tail -f $SLURM_SCRIPT"

    {
      echo "[$(date '+%F %T')] submitted job_id=$JOB_ID policy=$POLICY group=$GROUP"
      echo "  log=$JOB_LOG"
      echo "  slurm_script=$SLURM_SCRIPT"
    } >> "$SUMMARY_LOG"

    if [ "$FOLLOW_LOGS" = true ]; then
      follow_job_log "$JOB_ID" "$JOB_LOG"
    fi

    ((POLICY_NUM++))
  done

  ((GROUP_NUM++))
done

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}All jobs submitted in sequence.${NC}"
echo -e "${YELLOW}Summary log:${NC} tail -f $SUMMARY_LOG"
echo -e "${YELLOW}Queue:${NC}       squeue -u \$USER"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

