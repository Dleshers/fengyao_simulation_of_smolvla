#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-$ROOT/.conda/isaaclab}"
MODE="${1:-torque}"
CONTROL_MODE="${CONTROL_MODE:-joint}"
EXTRA=()
if [[ "$MODE" == "torque" ]]; then
  EXTRA+=(--send-gripper-torque-window --torque-window-size 30)
elif [[ "$MODE" != "visual" ]]; then
  echo "Usage: $0 [visual|torque]" >&2
  exit 2
fi

cd "$ROOT/IsaacLab-Tactile"
mkdir -p "$ROOT/tmp"
if [[ "$CONTROL_MODE" == "joint" ]]; then
  TASK="Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0"
else
  TASK="Isaac-Pick-Place-Basket-Franka-IK-Rel-TacEx-v0"
fi
OMNI_KIT_ACCEPT_EULA=YES TERM=xterm TMPDIR="$ROOT/tmp" "$ISAAC_ENV/bin/python" scripts/eval_server.py \
  --host='*' --port="${PORT:-5555}" \
  --env="$TASK" \
  --enable_cameras --headless \
  --experience=isaacsim_4_5/isaaclab.python.headless.rendering.kit \
  --kit_args=--/app/useFabricSceneDelegate=0 \
  ${TRAJECTORY_LOG:+--trajectory-log="$TRAJECTORY_LOG"} \
  "${EXTRA[@]}"
