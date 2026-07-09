#!/usr/bin/env bash
set -euo pipefail

# Matched control for the torque-LSTM injection ablation.
#
# This run starts from the successful targeted5 visual 50k checkpoint and
# continues for 5k steps on the same targeted5-clean dataset, but keeps
# use_torque_lstm=false.  Compare this against:
#
#   targeted5_visual50k_plus_torque_lstm_5k_seed1000_20260708
#
# If this visual-only continuation stays near the visual 50k success rate while
# the torque run drops, the drop is attributable to torque injection. If both
# drop similarly, the issue is more likely extra fine-tuning / dataset effects.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-targeted5_visual50k_plus_visualonly_5k_seed1000_20260708}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-$ROOT/_runtime/remote_handoff_gripper_lstm_work/persistent}"

export PRETRAINED_POLICY="${PRETRAINED_POLICY:-$PERSISTENT_ROOT/gripper_lstm_experiments/targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean}"
export DATASET_ROOT="${DATASET_ROOT:-$PERSISTENT_ROOT/datasets/franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSISTENT_ROOT/gripper_lstm_experiments}"
export STEPS="${STEPS:-5000}"
export SAVE_FREQ="${SAVE_FREQ:-1000}"
export LOG_FREQ="${LOG_FREQ:-100}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SEED="${SEED:-1000}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TMP_BASE="${TMP_BASE:-/tmp/svl}"

mkdir -p "$PERSISTENT_ROOT/logs"

echo "[visual5k-control] RUN_NAME=$RUN_NAME"
echo "[visual5k-control] PRETRAINED_POLICY=$PRETRAINED_POLICY"
echo "[visual5k-control] DATASET_ROOT=$DATASET_ROOT"

RUN_NAME="$RUN_NAME" bash experiment/train_targeted5_visual_50k.sh 2>&1 \
  | tee "$PERSISTENT_ROOT/logs/${RUN_NAME}.log"
