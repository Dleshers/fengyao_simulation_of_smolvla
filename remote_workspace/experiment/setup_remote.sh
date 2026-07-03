#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work}"
CONDA="${CONDA:-/cs/student/msc/rai/2025/fenzhang/miniconda3/bin/conda}"
ISAAC_ENV="$ROOT/.conda/isaaclab"
LEROBOT_ENV="$ROOT/.venv/lerobot"

if [[ ! -x "$ISAAC_ENV/bin/python" ]]; then
  "$CONDA" create -p "$ISAAC_ENV" python=3.10 pip -y
fi

"$ISAAC_ENV/bin/pip" install \
  isaacsim-rl==4.5.0.0 \
  isaacsim-extscache-kit==4.5.0.0 \
  isaacsim-extscache-kit-sdk==4.5.0.0 \
  isaacsim-extscache-physics==4.5.0.0 \
  --extra-index-url https://pypi.nvidia.com
"$ISAAC_ENV/bin/pip" install --force-reinstall \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$ISAAC_ENV/bin/pip" install \
  -e "$ROOT/IsaacLab-Tactile/source/isaaclab" \
  -e "$ROOT/IsaacLab-Tactile/source/isaaclab_assets" \
  -e "$ROOT/IsaacLab-Tactile/source/isaaclab_tasks" \
  -e "$ROOT/IsaacLab-Tactile/source/isaaclab_mimic" \
  -e "$ROOT/IsaacLab-Tactile/source/isaaclab_rl" \
  -e "$ROOT/TacEx/source/tacex" \
  -e "$ROOT/TacEx/source/tacex_assets" \
  -e "$ROOT/TacEx/source/tacex_tasks"

if [[ ! -x "$LEROBOT_ENV/bin/python" ]]; then
  "$ISAAC_ENV/bin/python" -m venv "$LEROBOT_ENV"
fi
"$LEROBOT_ENV/bin/pip" install --upgrade pip setuptools wheel
"$LEROBOT_ENV/bin/pip" install -e "$ROOT/lerobot-tactile[smolvla]"

echo "Accept the Isaac Sim EULA once with:"
echo "  printf 'yes\\n' | $ISAAC_ENV/bin/python -c 'import isaacsim'"
