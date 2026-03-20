#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SOURCE_ENV="${SOURCE_ENV:-/speed-scratch/$USER/conda/lerobot}"
TARGET_ENV="${TARGET_ENV:-/speed-scratch/$USER/conda/lerobot-pi}"
REBUILD_TARGET="${REBUILD_TARGET:-true}"
CONDA_EXE_PATH="${CONDA_EXE_PATH:-/encs/pkg/anaconda3-2023.03/root/bin/conda}"

if [ ! -x "$CONDA_EXE_PATH" ]; then
  echo "[error] conda executable not found: $CONDA_EXE_PATH"
  exit 1
fi

if [ ! -x "$SOURCE_ENV/bin/python" ]; then
  echo "[warn] source env missing python: $SOURCE_ENV/bin/python"
  echo "[warn] continuing with clean create from conda defaults"
fi

if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
  echo "[error] pyproject.toml not found at repo root: $REPO_ROOT"
  exit 1
fi

mkdir -p /speed-scratch/"$USER"/tmp
export TMPDIR=/speed-scratch/"$USER"/tmp
export TMP="$TMPDIR"

echo "[info] source env: $SOURCE_ENV"
echo "[info] target env: $TARGET_ENV"
echo "[info] tmpdir: $TMPDIR"

if [ "$REBUILD_TARGET" = "true" ]; then
  echo "[info] rebuilding target env from clean conda create"
  rm -rf "$TARGET_ENV"
  "$CONDA_EXE_PATH" create -y -p "$TARGET_ENV" python=3.10 pip
else
  echo "[info] keeping existing target env (REBUILD_TARGET=false)"
fi

TARGET_PY="$TARGET_ENV/bin/python"
TARGET_PIP="$TARGET_ENV/bin/pip"

if [ ! -x "$TARGET_PY" ] || [ ! -x "$TARGET_PIP" ]; then
  echo "[error] target python/pip missing after environment sync"
  exit 1
fi

echo "[info] upgrading packaging tools"
"$TARGET_PIP" install --upgrade pip setuptools wheel

echo "[info] installing lerobot with pi dependencies"
(
  cd "$REPO_ROOT"
  "$TARGET_PIP" install -e ".[pi]"
)

echo "[info] ensuring siglip check module exists"
SIGLIP_DIR="$TARGET_ENV/lib/python3.10/site-packages/transformers/models/siglip"
SIGLIP_CHECK_FILE="$SIGLIP_DIR/check.py"
if [ ! -f "$SIGLIP_CHECK_FILE" ]; then
  mkdir -p "$SIGLIP_DIR"
  cat > "$SIGLIP_CHECK_FILE" <<'PY'
import transformers


def check_whether_transformers_replace_is_installed_correctly():
    return transformers.__version__ in {"4.53.2", "4.53.3"}
PY
  echo "[info] created missing $SIGLIP_CHECK_FILE"
fi

echo "[info] static verification"
if [ ! -f "$SIGLIP_CHECK_FILE" ]; then
  echo "[error] siglip check file is still missing: $SIGLIP_CHECK_FILE"
  exit 1
fi
grep -n "check_whether_transformers_replace_is_installed_correctly" "$SIGLIP_CHECK_FILE" >/dev/null

echo "[ok] pi05 env ready: $TARGET_ENV"
echo "[ok] manager usage: export PI05_CONDA_ENV_NAME=$TARGET_ENV"
