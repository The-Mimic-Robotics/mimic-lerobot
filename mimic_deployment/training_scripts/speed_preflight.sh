#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_RESOLVER="$SCRIPT_DIR/dataset_groups.py"

DATASET_GROUP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-group)
      DATASET_GROUP="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --dataset-group <group_name>"
      exit 1
      ;;
  esac
done

if [ -z "$DATASET_GROUP" ]; then
  echo "Usage: $0 --dataset-group <group_name>"
  exit 1
fi

echo "[preflight] dataset-group=$DATASET_GROUP"

echo "[preflight] disk quota"
quota -s || true

echo "[preflight] gpu visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current --format=csv,noheader || true
else
  echo "nvidia-smi not found"
fi

echo "[preflight] python/env"
python - <<'PY'
import importlib
modules = ["torch", "accelerate", "wandb", "huggingface_hub"]
for m in modules:
    importlib.import_module(m)
print("python module check: ok")
PY

echo "[preflight] wandb connectivity"
python - <<'PY'
import wandb
api = wandb.Api(timeout=20)
viewer = api.viewer
entity = viewer.get("entity", "<unknown>") if isinstance(viewer, dict) else "<ok>"
print(f"wandb viewer ok: {entity}")
PY

echo "[preflight] huggingface datasets"
mapfile -t DATASET_LIST < <(python "$DATASET_RESOLVER" "$DATASET_GROUP" --format list)
if [ "${#DATASET_LIST[@]}" -eq 0 ]; then
  echo "No datasets resolved for group '$DATASET_GROUP'"
  exit 1
fi

for repo in "${DATASET_LIST[@]}"; do
  echo "  - checking $repo"
  python - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id = sys.argv[1]
api = HfApi(token=None)
info = api.dataset_info(repo_id=repo_id)
print(f"    ok: {info.id}")
PY
done

echo "[preflight] all checks passed"
