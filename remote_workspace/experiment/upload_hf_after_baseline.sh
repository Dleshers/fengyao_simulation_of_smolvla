#!/usr/bin/env bash
set -euo pipefail

PERSISTENT_ROOT="/cs/student/project_msc/2025/rai/fenzhang/simulation_storage"
RUN_DIR="$PERSISTENT_ROOT/gripper_lstm_experiments/baseline_smolvla_50000_seed1000"
FINAL_DIR="$RUN_DIR/checkpoints/050000/pretrained_model"
STATE_FILE="$RUN_DIR/checkpoints/050000/training_state/training_step.json"
TRAIN_LOG="$PERSISTENT_ROOT/logs/post_collection_baseline.log"
CARD="$PERSISTENT_ROOT/remote_handoff_gripper_lstm_workspace/experiment/hf_baseline_model_card.md"
HF="/cs/student/msc/rai/2025/fenzhang/.local/bin/hf"
REPO="Dleshers/smolvla-franka-pickplace-baseline-50k-seed1000"

echo "[$(date -Is)] Waiting for completed 50K baseline"
while ! grep -q 'PASS: 50000-step pure SmolVLA baseline completed' "$TRAIN_LOG" 2>/dev/null; do
  sleep 60
done

echo "[$(date -Is)] Completion marker found; validating final checkpoint"
test -s "$FINAL_DIR/model.safetensors"
test -s "$FINAL_DIR/config.json"
test -s "$FINAL_DIR/train_config.json"
test -s "$STATE_FILE"
grep -Eq '(^|[^0-9])50000([^0-9]|$)' "$STATE_FILE"
if grep -Eiq 'Traceback|CUDA out of memory|Killed|ERROR' "$TRAIN_LOG"; then
  echo "ERROR: fatal-looking text found in training log; refusing upload" >&2
  exit 3
fi

install -m 0644 "$CARD" "$FINAL_DIR/README.md"
"$HF" repos create "$REPO" --type model --private --exist-ok
"$HF" upload "$REPO" "$FINAL_DIR" --repo-type model --private \
  --commit-message "Upload completed 50K visual-only baseline"
"$HF" models info "$REPO" --format json
echo "[$(date -Is)] PASS: private Hugging Face model upload completed: $REPO"
