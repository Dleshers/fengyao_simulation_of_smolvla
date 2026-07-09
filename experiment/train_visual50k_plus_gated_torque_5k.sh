#!/usr/bin/env bash
set -euo pipefail

# Conservative torque-LSTM continuation from the successful targeted5 visual
# 50k checkpoint.  The torque adapter is zero-initialized, so the initial
# torque token is exactly zero and the initial policy is functionally close to
# the visual baseline instead of receiving a random torque suffix token.
#
# Keep the scalar gate nonzero (default: 1.0).  If both the adapter and gate are
# initialized to zero, the torque branch is dead: neither the gate nor adapter
# receives a useful gradient at the first step.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$RUNTIME_ROOT/lerobot-tactile}"
LEROBOT_ENV="${LEROBOT_ENV:-$RUNTIME_ROOT/.venv/lerobot}"

DATASET_REPO_ID="${DATASET_REPO_ID:-franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean}"
DATASET_ROOT="${DATASET_ROOT:-$RUNTIME_ROOT/persistent/datasets/$DATASET_REPO_ID}"
PRETRAINED_POLICY="${PRETRAINED_POLICY:-$RUNTIME_ROOT/persistent/gripper_lstm_experiments/targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model}"
TORQUE_LSTM_WEIGHTS="${TORQUE_LSTM_WEIGHTS:-$REPO_ROOT/trained_lstm_weights/torque_16d_encoder.pt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$RUNTIME_ROOT/persistent/gripper_lstm_experiments}"
RUN_NAME="${RUN_NAME:-targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_seed1000_20260709}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"

SEED="${SEED:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TORQUE_GATE_INIT="${TORQUE_GATE_INIT:-1.0}"

TMP_BASE="${TMP_BASE:-/tmp/svl}"
mkdir -p "$TMP_BASE" "$RUNTIME_ROOT/persistent/logs"

export TMPDIR="${TMPDIR:-$TMP_BASE}"
export TMP="${TMP:-$TMP_BASE}"
export TEMP="${TEMP:-$TMP_BASE}"
export HF_HOME="${HF_HOME:-$RUNTIME_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_ROOT/.cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ ! -x "$LEROBOT_ENV/bin/lerobot-train" ]]; then
  echo "Missing lerobot-train: $LEROBOT_ENV/bin/lerobot-train" >&2
  exit 2
fi
if [[ ! -d "$LEROBOT_ROOT" ]]; then
  echo "Missing LeRobot workspace: $LEROBOT_ROOT" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset root does not exist: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -d "$PRETRAINED_POLICY" ]]; then
  echo "Pretrained policy directory does not exist: $PRETRAINED_POLICY" >&2
  exit 2
fi
if [[ ! -f "$TORQUE_LSTM_WEIGHTS" ]]; then
  echo "Standalone torque LSTM weights do not exist: $TORQUE_LSTM_WEIGHTS" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

echo "Targeted5 gated torque-LSTM 5k continuation"
echo "  dataset.repo_id:      $DATASET_REPO_ID"
echo "  dataset.root:         $DATASET_ROOT"
echo "  policy.path:          $PRETRAINED_POLICY"
echo "  torque weights:       $TORQUE_LSTM_WEIGHTS"
echo "  zero-init adapter:    true"
echo "  torque gate init:     $TORQUE_GATE_INIT"
echo "  output_dir:           $OUTPUT_DIR"
echo "  steps/save_freq:      $STEPS / $SAVE_FREQ"
echo "  seed/batch:           $SEED / $BATCH_SIZE"

"$LEROBOT_ENV/bin/python" "$RUNTIME_ROOT/experiment/validate_dataset.py" \
  --repo-id "$DATASET_REPO_ID" \
  --root "$DATASET_ROOT" \
  --window-size 30 \
  --samples 128 \
  --sequence-checks 2048

cd "$LEROBOT_ROOT"

"$LEROBOT_ENV/bin/lerobot-train" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$PRETRAINED_POLICY" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_tactile=false \
  --policy.use_torque_lstm=true \
  --policy.torque_window_key=observation.gripper_torque \
  --policy.torque_window_size=30 \
  --policy.torque_input_dim=1 \
  --policy.torque_lstm_hidden_dim=32 \
  --policy.torque_lstm_output_dim=16 \
  --policy.torque_lstm_num_layers=1 \
  --policy.torque_lstm_weights_path="$TORQUE_LSTM_WEIGHTS" \
  --policy.train_torque_lstm=false \
  --policy.torque_zero_init_adapter=true \
  --policy.torque_gate_init="$TORQUE_GATE_INIT" \
  --seed="$SEED" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers="$NUM_WORKERS" \
  --log_freq="$LOG_FREQ" \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" 2>&1 \
  | tee "$RUNTIME_ROOT/persistent/logs/${RUN_NAME}.log"
