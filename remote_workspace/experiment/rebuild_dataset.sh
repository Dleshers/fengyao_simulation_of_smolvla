#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
ISAAC_ENV="${ISAAC_ENV:-$ROOT/.conda/isaaclab}"
LEROBOT_ENV="${LEROBOT_ENV:-$ROOT/.venv/lerobot}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/cs/student/project_msc/2025/rai/fenzhang/simulation_storage}"
RAW_DIR="${RAW_DIR:-$PERSISTENT_ROOT/datasets/franka_pickplace_joint_torque_w30_raw}"
DATASET_PARENT="${DATASET_PARENT:-$PERSISTENT_ROOT/datasets}"
DATASET_REPO_ID="${DATASET_REPO_ID:-franka_pickplace_joint_visual_torque_w30_v1}"
NUM_DEMOS="${NUM_DEMOS:-200}"
NUM_ENVS="${NUM_ENVS:-4}"
SEED="${SEED:-1000}"
PHASE="${1:-all}"

collect() {
  bash "$ROOT/experiment/preflight_collection.sh"
  if [[ -e "$RAW_DIR/data.hdf5" ]]; then
    echo "Refusing to overwrite existing raw dataset: $RAW_DIR/data.hdf5" >&2
    exit 2
  fi
  mkdir -p "$RAW_DIR" "$ROOT/tmp"
  cd "$ROOT/IsaacLab-Tactile"
  set +e
  OMNI_KIT_ACCEPT_EULA=YES TERM=xterm TMPDIR="$ROOT/tmp" \
    "$ISAAC_ENV/bin/python" scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
      --num_envs "$NUM_ENVS" --num_demos "$NUM_DEMOS" --seed "$SEED" \
      --save_demos --output_dir "$RAW_DIR" \
      --background_mode fixed --background_texture small_empty_house_4k.hdr \
      --enable_cameras --headless \
      --experience=isaacsim_4_5/isaaclab.python.headless.rendering.kit \
      --kit_args=--/app/useFabricSceneDelegate=0
  collection_status=$?
  set -e
  if [[ "$collection_status" -ne 0 && ! -s "$RAW_DIR/data.hdf5" ]]; then
    echo "ERROR: collection exited with status $collection_status before producing HDF5" >&2
    exit "$collection_status"
  elif [[ "$collection_status" -ne 0 ]]; then
    echo "WARNING: simulator cleanup exited with status $collection_status; auditing finalized HDF5"
  fi
  "$LEROBOT_ENV/bin/python" "$ROOT/experiment/inspect_raw_hdf5.py" "$RAW_DIR/data.hdf5"
}

convert() {
  local input="$RAW_DIR/data.hdf5"
  local output="$DATASET_PARENT/$DATASET_REPO_ID"
  if [[ ! -f "$input" ]]; then
    echo "Raw HDF5 does not exist: $input" >&2
    exit 2
  fi
  "$LEROBOT_ENV/bin/python" "$ROOT/experiment/inspect_raw_hdf5.py" "$input"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing LeRobot dataset: $output" >&2
    exit 2
  fi
  mkdir -p "$DATASET_PARENT"
  cd "$ROOT"
  "$LEROBOT_ENV/bin/python" IsaacLab-Tactile/lerobot/convert_pick_place_basket_joint_tacex.py \
    --input "$input" --output-dir "$DATASET_PARENT" --repo-id "$DATASET_REPO_ID" \
    --fps 20 --torque-window-size 30 --use-videos
  "$LEROBOT_ENV/bin/python" experiment/validate_dataset.py \
    --repo-id "$DATASET_REPO_ID" --root "$output" --window-size 30 --samples 64
}

case "$PHASE" in
  collect) collect ;;
  convert) convert ;;
  all) collect; convert ;;
  *) echo "Usage: $0 [collect|convert|all]" >&2; exit 2 ;;
esac
