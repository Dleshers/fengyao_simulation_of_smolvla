#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_REPO_ID:?Set DATASET_REPO_ID}"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${PRETRAINED_POLICY:?Set PRETRAINED_POLICY}"
: "${TORQUE_LSTM_WEIGHTS:?Set TORQUE_LSTM_WEIGHTS to the standalone encoder checkpoint}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset root does not exist: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -f "$TORQUE_LSTM_WEIGHTS" ]]; then
  echo "Standalone torque LSTM weights do not exist: $TORQUE_LSTM_WEIGHTS" >&2
  exit 2
fi
case "$PRETRAINED_POLICY" in
  *visual_050000*|*torque_lstm_030000*)
    echo "Refusing invalid legacy experiment checkpoint: $PRETRAINED_POLICY" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ENV="${LEROBOT_ENV:-$ROOT/.venv/lerobot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/gripper_lstm_experiments}"
STEPS="${STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-1000}"
RUN_TAG="${RUN_TAG:-smoke}"

COMMON=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --policy.path="$PRETRAINED_POLICY"
  --policy.device=cuda
  --policy.push_to_hub=false
  --seed="$SEED"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --save_freq="$STEPS"
)

"$LEROBOT_ENV/bin/python" "$ROOT/experiment/validate_dataset.py" \
  --repo-id "$DATASET_REPO_ID" --root "$DATASET_ROOT" --window-size 30

"$LEROBOT_ENV/bin/lerobot-train" "${COMMON[@]}" \
  --policy.use_torque_lstm=false \
  --output_dir="$OUTPUT_ROOT/${RUN_TAG}_visual_seed${SEED}"

"$LEROBOT_ENV/bin/lerobot-train" "${COMMON[@]}" \
  --policy.use_torque_lstm=true \
  --policy.torque_window_key=observation.gripper_torque \
  --policy.torque_window_size=30 \
  --policy.torque_input_dim=1 \
  --policy.torque_lstm_hidden_dim=32 \
  --policy.torque_lstm_output_dim=16 \
  --policy.torque_lstm_num_layers=1 \
  --policy.torque_lstm_weights_path="$TORQUE_LSTM_WEIGHTS" \
  --policy.train_torque_lstm=false \
  --output_dir="$OUTPUT_ROOT/${RUN_TAG}_torque_lstm_seed${SEED}"
