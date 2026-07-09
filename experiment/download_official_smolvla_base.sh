#!/usr/bin/env bash
set -euo pipefail

# Download the official Hugging Face / LeRobot SmolVLA pretrained checkpoint.
#
# This is the only allowed initialization for the next contact-rich task
# experiments.  Do not substitute the pick-and-place 50k checkpoints here:
# those are downstream task checkpoints, not official pretraining artifacts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
MODEL_REPO_ID="${MODEL_REPO_ID:-lerobot/smolvla_base}"
LOCAL_DIR="${LOCAL_DIR:-$RUNTIME_ROOT/pretrained/official_smolvla_base}"

export HF_HOME="${HF_HOME:-$RUNTIME_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_ROOT/.cache}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing hf CLI. Install current Hugging Face CLI first; do not use huggingface-cli." >&2
  exit 2
fi

echo "Checking Hugging Face authentication/account..."
hf auth whoami || true

if [[ -e "$LOCAL_DIR" && "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing official checkpoint directory: $LOCAL_DIR" >&2
  echo "Set FORCE_DOWNLOAD=1 only if you intentionally want hf to reconcile this directory." >&2
  exit 2
fi

mkdir -p "$LOCAL_DIR"

echo "Downloading official SmolVLA base model"
echo "  repo:      $MODEL_REPO_ID"
echo "  local dir: $LOCAL_DIR"
echo "  HF_HOME:   $HF_HOME"

hf download "$MODEL_REPO_ID" \
  --local-dir "$LOCAL_DIR"

test -s "$LOCAL_DIR/model.safetensors"
test -s "$LOCAL_DIR/config.json"

echo "Official SmolVLA base checkpoint restored: $LOCAL_DIR"

