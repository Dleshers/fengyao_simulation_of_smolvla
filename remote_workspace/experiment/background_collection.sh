#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work"
PERSISTENT_ROOT="/cs/student/project_msc/2025/rai/fenzhang/simulation_storage"
export TMPDIR="$WORK_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

echo "[$(date -Is)] Resuming environment setup"
if [[ -e "$WORK_ROOT/.environment_setup_complete" ]]; then
  echo "[$(date -Is)] Environment setup already completed; skipping installation"
else
  "$WORK_ROOT/.conda/isaaclab/bin/pip" install 'setuptools<81'
  "$WORK_ROOT/.conda/isaaclab/bin/pip" install --no-build-isolation 'flatdict==4.0.1'
  ROOT="$WORK_ROOT" bash "$WORK_ROOT/experiment/setup_remote.sh"
  touch "$WORK_ROOT/.environment_setup_complete"
fi

echo "[$(date -Is)] Running full collection preflight"
RUN_ISAAC_SMOKE=1 PERSISTENT_ROOT="$PERSISTENT_ROOT" \
  bash "$WORK_ROOT/experiment/preflight_collection.sh"

echo "[$(date -Is)] Running two-demo smoke collection"
PERSISTENT_ROOT="$PERSISTENT_ROOT" RUN_ISAAC_SMOKE=0 \
  bash "$WORK_ROOT/experiment/run_hdf5_smoke.sh"

echo "[$(date -Is)] Running full 200-demo collection"
PERSISTENT_ROOT="$PERSISTENT_ROOT" RUN_ISAAC_SMOKE=0 \
  bash "$WORK_ROOT/experiment/run_hdf5_full.sh"

echo "[$(date -Is)] Background setup, validation, and collection completed"
