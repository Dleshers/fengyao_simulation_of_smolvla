#!/usr/bin/env bash
set -euo pipefail

# Minimal contact-rich SmolVLA smoke training for the peg-insert/preinsert
# dataset family.  This is intentionally short by default; it validates that
# official SmolVLA pretraining can consume the new 49D/7D/two-camera/torque
# interface and that the torque-gated branch can train without immediately
# destabilizing the run.
#
# Arms:
#   visual    : official base + original dataset, use_torque_lstm=false
#   torque    : official base + original dataset, frozen torque-LSTM + gated zero-init adapter
#   zero      : same as torque, but dataset torque windows are all zeros
#   shuffle   : same as torque, but torque windows are episode-shuffled

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$RUNTIME_ROOT/lerobot-tactile}"
LEROBOT_ENV="${LEROBOT_ENV:-$RUNTIME_ROOT/.venv/lerobot}"

ARM="${ARM:-torque}"
case "$ARM" in
  visual)
    DATASET_REPO_ID="${DATASET_REPO_ID:-Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-v1}"
    USE_TORQUE_LSTM=false
    ;;
  torque)
    DATASET_REPO_ID="${DATASET_REPO_ID:-Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-v1}"
    USE_TORQUE_LSTM=true
    ;;
  zero)
    DATASET_REPO_ID="${DATASET_REPO_ID:-Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-zero-torque-v1}"
    USE_TORQUE_LSTM=true
    ;;
  shuffle)
    DATASET_REPO_ID="${DATASET_REPO_ID:-Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-shuffle-torque-v1}"
    USE_TORQUE_LSTM=true
    ;;
  *)
    echo "Unsupported ARM=$ARM; expected visual|torque|zero|shuffle" >&2
    exit 2
    ;;
esac

DATASET_ROOT="${DATASET_ROOT:-$RUNTIME_ROOT/persistent/lerobot_datasets/$DATASET_REPO_ID}"
PRETRAINED_POLICY="${PRETRAINED_POLICY:-$RUNTIME_ROOT/pretrained/official_smolvla_base}"
TORQUE_LSTM_WEIGHTS="${TORQUE_LSTM_WEIGHTS:-$REPO_ROOT/trained_lstm_weights/torque_16d_encoder.pt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$RUNTIME_ROOT/persistent/gripper_lstm_experiments}"
RUN_NAME="${RUN_NAME:-peg_insert_preinsert_${ARM}_mvp_smoke_steps${STEPS:-100}_seed${SEED:-1000}_20260710}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"

SEED="${SEED:-1000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
STEPS="${STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
LOG_FREQ="${LOG_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TORQUE_GATE_INIT="${TORQUE_GATE_INIT:-1.0}"
STATE_DIM="${STATE_DIM:-21}"
ACTION_DIM="${ACTION_DIM:-7}"
MAX_STATE_DIM="${MAX_STATE_DIM:-32}"

TMP_BASE="${TMP_BASE:-/tmp/svl}"
mkdir -p "$TMP_BASE" "$RUNTIME_ROOT/persistent/logs"

export TMPDIR="${TMPDIR:-$TMP_BASE}"
export TMP="${TMP:-$TMP_BASE}"
export TEMP="${TEMP:-$TMP_BASE}"
export HF_HOME="${HF_HOME:-$RUNTIME_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
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
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "Dataset meta/info.json missing: $DATASET_ROOT/meta/info.json" >&2
  exit 2
fi
if [[ ! -f "$PRETRAINED_POLICY/model.safetensors" ]]; then
  echo "Official pretrained model missing: $PRETRAINED_POLICY/model.safetensors" >&2
  exit 2
fi
if [[ "$USE_TORQUE_LSTM" == "true" && ! -f "$TORQUE_LSTM_WEIGHTS" ]]; then
  echo "Torque LSTM weights missing: $TORQUE_LSTM_WEIGHTS" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

echo "Peg-insert/preinsert MVP smoke training"
echo "  arm:              $ARM"
echo "  dataset.repo_id:  $DATASET_REPO_ID"
echo "  dataset.root:     $DATASET_ROOT"
echo "  policy.path:      $PRETRAINED_POLICY"
echo "  use_torque_lstm:  $USE_TORQUE_LSTM"
echo "  output_dir:       $OUTPUT_DIR"
echo "  steps/save_freq:  $STEPS / $SAVE_FREQ"
echo "  seed/batch:       $SEED / $BATCH_SIZE"

"$LEROBOT_ENV/bin/python" "$REPO_ROOT/experiment/audit_lerobot_dataset_features.py" \
  --repo-id "$DATASET_REPO_ID" \
  --root "$DATASET_ROOT" \
  --state-dim "$STATE_DIM" --action-dim "$ACTION_DIM"

cd "$LEROBOT_ROOT"

common_args=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --policy.path="$PRETRAINED_POLICY"
  --policy.device=cuda
  --policy.max_state_dim="$MAX_STATE_DIM"
  --policy.push_to_hub=false
  --policy.use_tactile=false
  --seed="$SEED"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --num_workers="$NUM_WORKERS"
  --log_freq="$LOG_FREQ"
  --save_freq="$SAVE_FREQ"
  --wandb.enable=false
  --output_dir="$OUTPUT_DIR"
)

if [[ "$USE_TORQUE_LSTM" == "true" ]]; then
  "$LEROBOT_ENV/bin/lerobot-train" \
    "${common_args[@]}" \
    --policy.use_torque_lstm=true \
    --policy.torque_window_key=observation.gripper_torque \
    --policy.torque_window_size=30 \
    --policy.torque_input_dim=1 \
    --policy.torque_lstm_hidden_dim=32 \
    --policy.torque_lstm_output_dim=16 \
    --policy.torque_lstm_num_layers=1 \
    --policy.torque_lstm_weights_path="$TORQUE_LSTM_WEIGHTS" \
    --policy.train_torque_lstm=false 2>&1 \
    | tee "$RUNTIME_ROOT/persistent/logs/${RUN_NAME}.log"
else
  "$LEROBOT_ENV/bin/lerobot-train" \
    "${common_args[@]}" \
    --policy.use_torque_lstm=false 2>&1 \
    | tee "$RUNTIME_ROOT/persistent/logs/${RUN_NAME}.log"
fi
