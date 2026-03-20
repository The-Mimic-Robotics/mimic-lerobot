#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE_STEPS="${SMOKE_STEPS:-1000}"
SMOKE_SAVE_FREQ="${SMOKE_SAVE_FREQ:-1000}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-${BATCH_SIZE:-12}}"

echo "[xvla-smoke1k] steps=${SMOKE_STEPS} save_freq=${SMOKE_SAVE_FREQ} batch=${SMOKE_BATCH_SIZE}"

if [ -n "${JOB_NAME:-}" ]; then
  export JOB_NAME="${JOB_NAME}_smoke1k"
fi

export XVLA_SPEED_MODE="default"
export BATCH_SIZE="${SMOKE_BATCH_SIZE}"
export STEPS="${SMOKE_STEPS}"
export SAVE_FREQ="${SMOKE_SAVE_FREQ}"

exec "$SCRIPT_DIR/train.sh" "$@"
