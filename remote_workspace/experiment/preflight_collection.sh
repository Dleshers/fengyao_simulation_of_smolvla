#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-$ROOT/.conda/isaaclab}"
LEROBOT_ENV="${LEROBOT_ENV:-$ROOT/.venv/lerobot}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/cs/student/project_msc/2025/rai/fenzhang/simulation_storage}"
RAW_DIR="${RAW_DIR:-$PERSISTENT_ROOT/datasets/franka_pickplace_joint_torque_w30_raw}"

required_files=(
  "$ROOT/IsaacLab-Tactile/scripts/environments/state_machine/pick_place_basket_tacex_sm.py"
  "$ROOT/IsaacLab-Tactile/lerobot/convert_pick_place_basket_joint_tacex.py"
  "$ROOT/experiment/inspect_raw_hdf5.py"
  "$ROOT/experiment/validate_dataset.py"
  "$ROOT/lerobot-tactile/src/lerobot/policies/factory.py"
  "$ROOT/lerobot-tactile/src/lerobot/policies/smolvla/modeling_smolvla.py"
)

for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
[[ -x "$ISAAC_ENV/bin/python" ]] || { echo "ERROR: missing Isaac Python: $ISAAC_ENV/bin/python" >&2; exit 2; }
[[ -x "$LEROBOT_ENV/bin/python" ]] || { echo "ERROR: missing LeRobot Python: $LEROBOT_ENV/bin/python" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is unavailable" >&2; exit 2; }

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== Python environments =="
"$ISAAC_ENV/bin/python" -c \
  'import torch; assert torch.cuda.is_available(); print("Isaac torch", torch.__version__, torch.cuda.get_device_name(0))'
"$LEROBOT_ENV/bin/python" -c \
  'import h5py, pytest, torch, lerobot; print("LeRobot torch", torch.__version__, "lerobot", lerobot.__file__)'

echo "== Storage =="
mkdir -p "$(dirname "$RAW_DIR")" "$ROOT/tmp"
df -h "$ROOT" "$(dirname "$RAW_DIR")"
if [[ -e "$RAW_DIR/data.hdf5" ]]; then
  echo "ERROR: target already exists and will not be overwritten: $RAW_DIR/data.hdf5" >&2
  exit 2
fi

echo "== Static source checks =="
grep -q 'gripper_torque' "$ROOT/IsaacLab-Tactile/scripts/environments/state_machine/pick_place_basket_tacex_sm.py"
grep -q 'torque-window-size' "$ROOT/IsaacLab-Tactile/lerobot/convert_pick_place_basket_joint_tacex.py"
grep -q 'dataset_input_features.pop("observation.gripper_torque"' \
  "$ROOT/lerobot-tactile/src/lerobot/policies/factory.py"
grep -q 'def prepare_torque_window' \
  "$ROOT/lerobot-tactile/src/lerobot/policies/smolvla/modeling_smolvla.py"

if [[ "${RUN_ISAAC_SMOKE:-0}" == "1" ]]; then
  echo "== Isaac headless smoke =="
  OMNI_KIT_ACCEPT_EULA=YES TERM=xterm TMPDIR="$ROOT/tmp" \
    "$ISAAC_ENV/bin/python" "$ROOT/IsaacLab-Tactile/scripts/tutorials/00_sim/create_empty.py" --headless
else
  echo "Skipping Isaac launch smoke; set RUN_ISAAC_SMOKE=1 to enable it."
fi

echo "PASS: collection preflight completed."
