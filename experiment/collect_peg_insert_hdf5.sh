#!/usr/bin/env bash
set -euo pipefail

# Record a small peg-insertion raw HDF5 demonstration set.
#
# Default mode is keyboard teleoperation because it is always available.
# For better demonstrations, run with TELEOP_DEVICE=spacemouse after the
# SpaceMouse device permissions are confirmed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
ISAAC_ROOT="${ISAAC_ROOT:-$RUNTIME_ROOT/IsaacLab-Tactile}"
ISAAC_ENV="${ISAAC_ENV:-$RUNTIME_ROOT/.conda/isaaclab}"

TASK="${TASK:-Isaac-Peg-Insert-Franka-IK-Rel-v0}"
TELEOP_DEVICE="${TELEOP_DEVICE:-keyboard}"
NUM_DEMOS="${NUM_DEMOS:-10}"
STEP_HZ="${STEP_HZ:-20}"
NUM_SUCCESS_STEPS="${NUM_SUCCESS_STEPS:-10}"
PEG_INSERT_PROCEDURAL_ASSETS="${PEG_INSERT_PROCEDURAL_ASSETS:-1}"
PEG_INSERT_SIMPLE_TABLE="${PEG_INSERT_SIMPLE_TABLE:-1}"
OUT_DIR="${OUT_DIR:-$RUNTIME_ROOT/persistent/raw_hdf5/peg_insert_official_pretrain_smoke_$(date +%Y%m%d_%H%M%S)}"
DATASET_FILE="${DATASET_FILE:-$OUT_DIR/peg_insert_demos.hdf5}"

TMP_BASE="${TMP_BASE:-/tmp/svl}"
mkdir -p "$TMP_BASE" "$OUT_DIR" "$RUNTIME_ROOT/persistent/logs"

export TMPDIR="${TMPDIR:-$TMP_BASE}"
export TMP="${TMP:-$TMP_BASE}"
export TEMP="${TEMP:-$TMP_BASE}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PEG_INSERT_PROCEDURAL_ASSETS
export PEG_INSERT_SIMPLE_TABLE
export TERM="${TERM:-xterm}"
if [[ "$TERM" == "dumb" ]]; then
  export TERM=xterm
fi

if [[ ! -x "$ISAAC_ROOT/isaaclab.sh" ]]; then
  echo "Missing IsaacLab launcher: $ISAAC_ROOT/isaaclab.sh" >&2
  exit 2
fi
if [[ ! -x "$ISAAC_ENV/bin/python" ]]; then
  echo "Missing IsaacLab conda python: $ISAAC_ENV/bin/python" >&2
  exit 2
fi
if [[ -s "$DATASET_FILE" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing HDF5 file: $DATASET_FILE" >&2
  exit 2
fi

export CONDA_PREFIX="$ISAAC_ENV"
export PATH="$ISAAC_ENV/bin:$PATH"

echo "Peg-insert HDF5 collection"
echo "  task:          $TASK"
echo "  teleop device: $TELEOP_DEVICE"
echo "  num demos:     $NUM_DEMOS"
echo "  step Hz:       $STEP_HZ"
echo "  dataset file:  $DATASET_FILE"

cd "$ISAAC_ROOT"

./isaaclab.sh -p scripts/tools/record_demos.py \
  --task "$TASK" \
  --teleop_device "$TELEOP_DEVICE" \
  --dataset_file "$DATASET_FILE" \
  --step_hz "$STEP_HZ" \
  --num_demos "$NUM_DEMOS" \
  --num_success_steps "$NUM_SUCCESS_STEPS" \
  --enable_cameras \
  2>&1 | tee "$RUNTIME_ROOT/persistent/logs/peg_insert_hdf5_collection.log"
