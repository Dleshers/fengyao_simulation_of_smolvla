#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${LEROBOT_ROOT:-/home/radu/SmolVLA-Fengyao/lerobot}"
HANDOFF="${HANDOFF_ROOT:-/home/radu/SmolVLA-Fengyao/remote_handoff_gripper_lstm}"
CONFIG_DST="$ROOT/src/lerobot/policies/smolvla/configuration_smolvla.py"
MODEL_DST="$ROOT/src/lerobot/policies/smolvla/modeling_smolvla.py"
TEST_DST="$ROOT/tests/policies/smolvla/test_smolvla_torque_lstm.py"

for source in \
  "$HANDOFF/lerobot_overrides/configuration_smolvla.py" \
  "$HANDOFF/lerobot_overrides/modeling_smolvla.py" \
  "$HANDOFF/lerobot_overrides/test_smolvla_torque_lstm.py"; do
  [[ -f "$source" ]] || { echo "ERROR: missing $source" >&2; exit 2; }
done

mkdir -p "$(dirname -- "$TEST_DST")"
for target in "$CONFIG_DST" "$MODEL_DST"; do
  backup="$target.before_gripper_lstm"
  if [[ ! -e "$backup" ]]; then
    cp --preserve=mode,timestamps "$target" "$backup"
    echo "Backup: $backup"
  fi
done

cp "$HANDOFF/lerobot_overrides/configuration_smolvla.py" "$CONFIG_DST"
cp "$HANDOFF/lerobot_overrides/modeling_smolvla.py" "$MODEL_DST"
cp "$HANDOFF/lerobot_overrides/test_smolvla_torque_lstm.py" "$TEST_DST"

source /home/radu/miniconda3/etc/profile.d/conda.sh
conda activate smolvla_env_fengyao
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m py_compile "$CONFIG_DST" "$MODEL_DST" "$TEST_DST"
if python -c 'import pytest' >/dev/null 2>&1; then
  python -m pytest -q "$TEST_DST"
else
  echo "pytest is not installed; running equivalent dependency-free smoke checks."
  python - <<'PY'
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import ForceLSTMEncoder

encoder = ForceLSTMEncoder(input_dim=1, hidden_dim=32, output_dim=16, num_layers=1)
torque = torch.randn(4, 30, 1)
latent = encoder(torque)
latent.square().mean().backward()
assert latent.shape == (4, 16)
assert encoder.lstm.weight_ih_l0.grad is not None
assert encoder.fc.weight.grad is not None

baseline = SmolVLAConfig(use_torque_lstm=False)
baseline.input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
    "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
}
baseline.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
baseline.validate_features()

torque_cfg = SmolVLAConfig(
    use_torque_lstm=True,
    torque_window_size=30,
    torque_input_dim=1,
    torque_lstm_hidden_dim=32,
    torque_lstm_num_layers=1,
)
torque_cfg.input_features = {
    **baseline.input_features,
    "observation.gripper_torque": PolicyFeature(type=FeatureType.ENV, shape=(30, 1)),
}
torque_cfg.output_features = baseline.output_features
torque_cfg.validate_features()
print("PASS: dependency-free Torque-LSTM checks")
PY
fi
echo "PASS: Action-Expert-suffix Torque-LSTM override installed and tested."
