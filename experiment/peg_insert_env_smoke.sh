#!/usr/bin/env bash
set -euo pipefail

# Headless smoke test for the manager-based peg insertion task.
#
# This intentionally runs only a short finite zero-action rollout to verify task
# registration, Isaac headless/Vulkan startup, observation/action spaces, and
# basic step/reset.  It does not collect data, train, or evaluate a policy.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
ISAAC_ENV="${ISAAC_ENV:-$RUNTIME_ROOT/.conda/isaaclab}"

TASK="${TASK:-Isaac-Peg-Insert-Franka-IK-Rel-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
NUM_STEPS="${NUM_STEPS:-16}"
DUMP_AFTER_S="${DUMP_AFTER_S:-60}"
ACTION_MODE="${ACTION_MODE:-zero}"
ACTION_SCALE="${ACTION_SCALE:-0.01}"
DEVICE="${DEVICE:-cuda:0}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
PEG_INSERT_DISABLE_CAMERAS="${PEG_INSERT_DISABLE_CAMERAS:-$([[ "${ENABLE_CAMERAS:-1}" == "1" ]] && echo 0 || echo 1)}"
PEG_INSERT_PROCEDURAL_ASSETS="${PEG_INSERT_PROCEDURAL_ASSETS:-1}"
PEG_INSERT_SIMPLE_TABLE="${PEG_INSERT_SIMPLE_TABLE:-1}"

TMP_BASE="${TMP_BASE:-/tmp/svl}"
mkdir -p "$TMP_BASE" "$RUNTIME_ROOT/persistent/logs"

export TMPDIR="${TMPDIR:-$TMP_BASE}"
export TMP="${TMP:-$TMP_BASE}"
export TEMP="${TEMP:-$TMP_BASE}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PEG_INSERT_DISABLE_CAMERAS
export PEG_INSERT_PROCEDURAL_ASSETS
export PEG_INSERT_SIMPLE_TABLE
export TERM="${TERM:-xterm}"
if [[ "$TERM" == "dumb" ]]; then
  export TERM=xterm
fi

if [[ ! -x "$ISAAC_ENV/bin/python" ]]; then
  echo "Missing IsaacLab conda python: $ISAAC_ENV/bin/python" >&2
  exit 2
fi

export CONDA_PREFIX="$ISAAC_ENV"
export PATH="$ISAAC_ENV/bin:$PATH"

echo "Peg-insert Isaac headless smoke"
echo "  task:     $TASK"
echo "  num_envs: $NUM_ENVS"
echo "  steps:    $NUM_STEPS"
echo "  dump_s:   $DUMP_AFTER_S"
echo "  action:   $ACTION_MODE scale=$ACTION_SCALE"
echo "  cameras:  $ENABLE_CAMERAS"
echo "  cam_cfg:  $([[ "$PEG_INSERT_DISABLE_CAMERAS" == "1" ]] && echo disabled || echo enabled)"
echo "  assets:   $([[ "$PEG_INSERT_PROCEDURAL_ASSETS" == "1" ]] && echo procedural || echo usd)"
echo "  table:    $([[ "$PEG_INSERT_SIMPLE_TABLE" == "1" ]] && echo simple || echo usd)"
echo "  seconds:  $TIMEOUT_SECONDS"
echo "  device:   $DEVICE"
echo "  TMPDIR:   $TMPDIR"

LOG_PATH="$RUNTIME_ROOT/persistent/logs/peg_insert_env_smoke.log"

cmd=(
  "$ISAAC_ENV/bin/python" "$REPO_ROOT/experiment/peg_insert_headless_probe.py"
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --num_steps "$NUM_STEPS"
  --dump_after_s "$DUMP_AFTER_S"
  --action_mode "$ACTION_MODE"
  --action_scale "$ACTION_SCALE"
  --device "$DEVICE"
  --headless
  --experience=isaacsim_4_5/isaaclab.python.headless.rendering.kit
)
if [[ "$ENABLE_CAMERAS" == "1" ]]; then
  cmd+=(--enable_cameras)
fi

set +e
timeout "$TIMEOUT_SECONDS" "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
status=${PIPESTATUS[0]}
set -e

if grep -q "\[PEG_PROBE\] success" "$LOG_PATH"; then
  echo "Peg-insert finite probe completed successfully."
  exit 0
fi

if [[ "$status" == "124" ]]; then
  echo "Peg-insert finite probe timed out before reporting success." >&2
  exit 124
fi

exit "$status"
