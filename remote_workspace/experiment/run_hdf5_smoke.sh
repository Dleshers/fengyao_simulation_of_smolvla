#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/cs/student/project_msc/2025/rai/fenzhang/simulation_storage}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RAW_DIR="${RAW_DIR:-$PERSISTENT_ROOT/datasets/smoke_joint_torque_w30_${RUN_ID}}"
export NUM_DEMOS="${NUM_DEMOS:-2}"
export NUM_ENVS="${NUM_ENVS:-1}"
export SEED="${SEED:-1000}"
export RUN_ISAAC_SMOKE="${RUN_ISAAC_SMOKE:-1}"

mkdir -p "$RAW_DIR"
LOG="$RAW_DIR/collection.log"

echo "Smoke collection output: $RAW_DIR"
bash "$ROOT/experiment/rebuild_dataset.sh" collect 2>&1 | tee "$LOG"

test -s "$RAW_DIR/data.hdf5"
sha256sum "$RAW_DIR/data.hdf5" | tee "$RAW_DIR/data.hdf5.sha256"
echo "PASS: smoke HDF5 is ready: $RAW_DIR/data.hdf5"
echo "Next: inspect $LOG, then run the full collection script."
