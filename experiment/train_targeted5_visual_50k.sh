#!/usr/bin/env bash
set -euo pipefail

# Targeted5 visual-only SmolVLA 50k retraining/fine-tuning run.
#
# This script intentionally does not start torque-LSTM training.  It is meant
# to validate whether the targeted5 data augmentation that produced a 9/10
# short-run eval can hold up at the 50k scale.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$RUNTIME_ROOT/lerobot-tactile}"
LEROBOT_ENV="${LEROBOT_ENV:-$RUNTIME_ROOT/.venv/lerobot}"

DATASET_REPO_ID="${DATASET_REPO_ID:-franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean}"
DATASET_ROOT="${DATASET_ROOT:-$RUNTIME_ROOT/persistent/datasets/$DATASET_REPO_ID}"
PRETRAINED_POLICY="${PRETRAINED_POLICY:-$RUNTIME_ROOT/pretrained/baseline_smolvla_50k_seed1000}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$RUNTIME_ROOT/persistent/gripper_lstm_experiments}"
RUN_NAME="${RUN_NAME:-targeted5_visual_50k_seed1000_20260708}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"

SEED="${SEED:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RESUME="${RESUME:-0}"

TMP_BASE="${TMP_BASE:-/tmp/svl}"
mkdir -p "$TMP_BASE"

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
if [[ ! -f "$RUNTIME_ROOT/experiment/validate_dataset.py" ]]; then
  echo "Missing dataset validator: $RUNTIME_ROOT/experiment/validate_dataset.py" >&2
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
if [[ ! -d "$PRETRAINED_POLICY" ]]; then
  echo "Pretrained policy directory does not exist: $PRETRAINED_POLICY" >&2
  exit 2
fi
if [[ ! -f "$PRETRAINED_POLICY/model.safetensors" ]]; then
  echo "Pretrained policy model.safetensors missing: $PRETRAINED_POLICY/model.safetensors" >&2
  exit 2
fi
case "$PRETRAINED_POLICY" in
  *visual_050000*|*torque_lstm_030000*)
    echo "Refusing invalid legacy experiment checkpoint: $PRETRAINED_POLICY" >&2
    exit 2
    ;;
esac
if [[ "$RESUME" != "1" && -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ "$RESUME" == "1" && ! -d "$OUTPUT_DIR/checkpoints" ]]; then
  echo "Cannot resume: no checkpoints directory under $OUTPUT_DIR" >&2
  exit 2
fi

echo "Targeted5 visual-only 50k training"
echo "  dataset.repo_id: $DATASET_REPO_ID"
echo "  dataset.root:    $DATASET_ROOT"
echo "  policy.path:     $PRETRAINED_POLICY"
echo "  output_dir:      $OUTPUT_DIR"
echo "  steps/save_freq: $STEPS / $SAVE_FREQ"
echo "  seed/batch:      $SEED / $BATCH_SIZE"
echo "  num_workers:     $NUM_WORKERS"
echo "  TMPDIR:          $TMPDIR"
echo "  HF_HOME:         $HF_HOME"
echo "  resume:          $RESUME"

"$LEROBOT_ENV/bin/python" "$RUNTIME_ROOT/experiment/validate_dataset.py" \
  --repo-id "$DATASET_REPO_ID" \
  --root "$DATASET_ROOT" \
  --window-size 30 \
  --samples 128 \
  --sequence-checks 2048

cd "$LEROBOT_ROOT"

if [[ "$RESUME" == "1" ]]; then
  LATEST_CHECKPOINT="$(find "$OUTPUT_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
  CONFIG_PATH="$OUTPUT_DIR/checkpoints/$LATEST_CHECKPOINT/pretrained_model/train_config.json"
  if [[ ! -s "$CONFIG_PATH" ]]; then
    echo "Cannot resume: missing train config $CONFIG_PATH" >&2
    exit 2
  fi
  echo "Resuming from checkpoint $LATEST_CHECKPOINT"
  exec "$LEROBOT_ENV/bin/lerobot-train" \
    --config_path="$CONFIG_PATH" \
    --resume=true
fi

"$LEROBOT_ENV/bin/lerobot-train" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$PRETRAINED_POLICY" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_tactile=false \
  --policy.use_torque_lstm=false \
  --seed="$SEED" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers="$NUM_WORKERS" \
  --log_freq="$LOG_FREQ" \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR"
