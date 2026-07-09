#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
RUN_NAME="${RUN_NAME:-targeted5_visual_50k_seed1000_20260708}"
RUN_DIR="${RUN_DIR:-$RUNTIME_ROOT/persistent/gripper_lstm_experiments/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$RUNTIME_ROOT/persistent/logs/targeted5_visual_50k_resume_10000_20260708.log}"
INTERVAL="${INTERVAL:-10}"
TAIL_LINES="${TAIL_LINES:-12}"

while true; do
  clear || true
  printf 'Targeted5 visual 50k monitor — %s\n\n' "$(date -Is)"

  echo '[GPU]'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>&1 || true
  else
    echo 'nvidia-smi not found'
  fi

  echo
  echo '[Training process]'
  pgrep -af 'lerobot-train.*targeted5|lerobot-train.*010000/pretrained_model/train_config.json' || \
    echo 'No matching lerobot-train process found'

  echo
  echo '[Latest checkpoint]'
  latest=''
  if [[ -d "$RUN_DIR/checkpoints" ]]; then
    latest="$(find "$RUN_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
  fi
  if [[ -n "$latest" ]]; then
    state="$RUN_DIR/checkpoints/$latest/training_state/training_step.json"
    printf 'checkpoint: %s\n' "$latest"
    if [[ -s "$state" ]]; then
      printf 'state: '
      cat "$state"
      echo
    fi
  else
    echo 'No checkpoint found'
  fi

  echo
  printf '[Last %s log lines]\n' "$TAIL_LINES"
  if [[ -f "$LOG_FILE" ]]; then
    tail -n "$TAIL_LINES" "$LOG_FILE"
  else
    echo "Log not found: $LOG_FILE"
  fi

  echo
  printf 'Refresh: %ss — press Ctrl-C to exit\n' "$INTERVAL"
  sleep "$INTERVAL"
done
