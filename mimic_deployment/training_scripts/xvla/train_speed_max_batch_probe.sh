#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$REPO_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

BATCH_CANDIDATES="${BATCH_CANDIDATES:-64,56,48,40,32,28,24,20,16,12,8,6,4}"
PROBE_STEPS="${PROBE_STEPS:-120}"
PROBE_SAVE_FREQ="${PROBE_SAVE_FREQ:-1000}"
RUN_FINAL_AFTER_PROBE="${RUN_FINAL_AFTER_PROBE:-true}"
FINAL_STEPS="${FINAL_STEPS:-1000}"
FINAL_SAVE_FREQ="${FINAL_SAVE_FREQ:-1000}"
FINAL_BATCH_SIZE="${FINAL_BATCH_SIZE:-}"

BASE_JOB_NAME="${JOB_NAME:-xvla_speed_probe_$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a CANDIDATE_ARRAY <<< "$BATCH_CANDIDATES"
MAX_WORKING_BATCH=""

echo "[xvla-maxbatch] candidates=${BATCH_CANDIDATES} probe_steps=${PROBE_STEPS}"

for raw_batch in "${CANDIDATE_ARRAY[@]}"; do
  batch="$(echo "$raw_batch" | xargs)"
  if ! [[ "$batch" =~ ^[0-9]+$ ]] || [ "$batch" -le 0 ]; then
    echo "[xvla-maxbatch] skipping invalid candidate '$raw_batch'"
    continue
  fi

  probe_job_name="${BASE_JOB_NAME}_probe_b${batch}"
  probe_log="$LOG_DIR/${probe_job_name}.probe.log"

  echo "[xvla-maxbatch] probing batch=${batch} -> $probe_log"

  set +e
  (
    export XVLA_SPEED_MODE="default"
    export JOB_NAME="$probe_job_name"
    export BATCH_SIZE="$batch"
    export STEPS="$PROBE_STEPS"
    export SAVE_FREQ="$PROBE_SAVE_FREQ"
    "$SCRIPT_DIR/train.sh" --no-daemon
  ) >"$probe_log" 2>&1
  rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    MAX_WORKING_BATCH="$batch"
    echo "[xvla-maxbatch] success at batch=${batch}"
    break
  fi

  if grep -qiE 'torch\.OutOfMemoryError|CUDA out of memory|out of memory' "$probe_log"; then
    echo "[xvla-maxbatch] OOM at batch=${batch}, trying next candidate"
    continue
  fi

  echo "[xvla-maxbatch] non-OOM failure at batch=${batch}; stopping"
  tail -n 80 "$probe_log" || true
  exit "$rc"
done

if [ -z "$MAX_WORKING_BATCH" ]; then
  echo "[xvla-maxbatch] no working batch found in candidates: $BATCH_CANDIDATES"
  exit 1
fi

if [ -n "$FINAL_BATCH_SIZE" ]; then
  MAX_WORKING_BATCH="$FINAL_BATCH_SIZE"
fi

echo "[xvla-maxbatch] selected_batch=${MAX_WORKING_BATCH}"

if [ "$RUN_FINAL_AFTER_PROBE" != "true" ]; then
  echo "[xvla-maxbatch] probe-only mode complete"
  exit 0
fi

final_job_name="${BASE_JOB_NAME}_maxb${MAX_WORKING_BATCH}_${FINAL_STEPS}steps"

echo "[xvla-maxbatch] starting final validation run batch=${MAX_WORKING_BATCH} steps=${FINAL_STEPS}"

export XVLA_SPEED_MODE="default"
export JOB_NAME="$final_job_name"
export BATCH_SIZE="$MAX_WORKING_BATCH"
export STEPS="$FINAL_STEPS"
export SAVE_FREQ="$FINAL_SAVE_FREQ"

exec "$SCRIPT_DIR/train.sh" --no-daemon
