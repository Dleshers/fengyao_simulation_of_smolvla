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
DEVICE="${DEVICE:-cuda:0}"
HEADLESS="${HEADLESS:-0}"
EXPERIENCE="${EXPERIENCE:-isaacsim_4_5/isaaclab.python.rendering.kit}"
HEADLESS_EXPERIENCE="${HEADLESS_EXPERIENCE:-isaacsim_4_5/isaaclab.python.headless.rendering.kit}"
GEOMETRY_MODE="${GEOMETRY_MODE:-usd}"
LOCAL_ISAAC_4_5_ASSET_ROOT="${LOCAL_ISAAC_4_5_ASSET_ROOT:-$RUNTIME_ROOT/persistent/assets/isaac_4_5}"
PEG_INSERT_DISABLE_CAMERAS="${PEG_INSERT_DISABLE_CAMERAS:-0}"
OUT_DIR="${OUT_DIR:-$RUNTIME_ROOT/persistent/raw_hdf5/peg_insert_official_pretrain_smoke_$(date +%Y%m%d_%H%M%S)}"
DATASET_FILE="${DATASET_FILE:-$OUT_DIR/peg_insert_demos.hdf5}"

case "$GEOMETRY_MODE" in
  usd)
    PEG_INSERT_PROCEDURAL_ASSETS="${PEG_INSERT_PROCEDURAL_ASSETS:-0}"
    PEG_INSERT_SIMPLE_TABLE="${PEG_INSERT_SIMPLE_TABLE:-0}"
    ;;
  procedural)
    PEG_INSERT_PROCEDURAL_ASSETS="${PEG_INSERT_PROCEDURAL_ASSETS:-1}"
    PEG_INSERT_SIMPLE_TABLE="${PEG_INSERT_SIMPLE_TABLE:-1}"
    ;;
  *)
    echo "GEOMETRY_MODE must be 'usd' or 'procedural', got: $GEOMETRY_MODE" >&2
    exit 2
    ;;
esac

TMP_BASE="${TMP_BASE:-/tmp/svl}"
mkdir -p "$TMP_BASE" "$OUT_DIR" "$RUNTIME_ROOT/persistent/logs"

export TMPDIR="${TMPDIR:-$TMP_BASE}"
export TMP="${TMP:-$TMP_BASE}"
export TEMP="${TEMP:-$TMP_BASE}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export LOCAL_ISAAC_4_5_ASSET_ROOT
export PEG_INSERT_DISABLE_CAMERAS
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
echo "  geometry:      $GEOMETRY_MODE"
echo "  asset root:    $LOCAL_ISAAC_4_5_ASSET_ROOT"
echo "  cameras:       $([[ "$PEG_INSERT_DISABLE_CAMERAS" == "1" ]] && echo disabled || echo enabled)"
echo "  device:        $DEVICE"
echo "  headless:      $HEADLESS"
echo "  experience:    $([[ "$HEADLESS" == "1" ]] && echo "$HEADLESS_EXPERIENCE" || echo "$EXPERIENCE")"
echo "  dataset file:  $DATASET_FILE"

cd "$ISAAC_ROOT"

cmd=(
  "$ISAAC_ENV/bin/python" "$REPO_ROOT/experiment/record_peg_insert_demos.py"
  --task "$TASK"
  --teleop_device "$TELEOP_DEVICE"
  --dataset_file "$DATASET_FILE"
  --step_hz "$STEP_HZ"
  --num_demos "$NUM_DEMOS"
  --num_success_steps "$NUM_SUCCESS_STEPS"
  --max_steps_per_demo "${MAX_STEPS_PER_DEMO:-400}"
  --device "$DEVICE"
  --enable_cameras
)
if [[ "$HEADLESS" == "1" ]]; then
  cmd+=(--headless --experience "$HEADLESS_EXPERIENCE")
else
  cmd+=(--experience "$EXPERIENCE")
fi

"${cmd[@]}" \
  2>&1 | tee "$RUNTIME_ROOT/persistent/logs/peg_insert_hdf5_collection.log"
