#!/usr/bin/env bash
set -euo pipefail

HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$WORK_ROOT/lerobot-tactile}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$WORK_ROOT/IsaacLab-Tactile}"

[[ -d "$LEROBOT_ROOT/.git" ]] || { echo "ERROR: missing LeRobot clone: $LEROBOT_ROOT" >&2; exit 2; }
[[ -d "$ISAACLAB_ROOT/.git" ]] || { echo "ERROR: missing IsaacLab clone: $ISAACLAB_ROOT" >&2; exit 2; }

apply_once() {
  local repo="$1" patch="$2"
  if git -C "$repo" apply --check "$patch"; then
    git -C "$repo" apply "$patch"
    echo "Applied: $patch"
  elif git -C "$repo" apply --reverse --check "$patch"; then
    echo "Already applied: $patch"
  else
    echo "ERROR: patch is neither applicable nor already applied: $patch" >&2
    exit 2
  fi
}

CONFIG_DST="$LEROBOT_ROOT/src/lerobot/policies/smolvla/configuration_smolvla.py"
MODEL_DST="$LEROBOT_ROOT/src/lerobot/policies/smolvla/modeling_smolvla.py"
TEST_DST="$LEROBOT_ROOT/tests/policies/smolvla/test_smolvla_torque_lstm.py"
for target in "$CONFIG_DST" "$MODEL_DST"; do
  if [[ ! -e "$target.before_gripper_lstm_handoff" ]]; then
    cp --preserve=mode,timestamps "$target" "$target.before_gripper_lstm_handoff"
  fi
done

STAMP="$WORK_ROOT/.gripper_lstm_handoff_patches_applied"
if [[ ! -e "$STAMP" ]]; then
  if grep -q 'include_gripper_torque_window' "$LEROBOT_ROOT/src/lerobot/envs/configs.py"; then
    echo "Detected existing LeRobot remote-environment changes."
  else
    apply_once "$LEROBOT_ROOT" "$HANDOFF_ROOT/patches/remote_workspace_lerobot.patch"
  fi
  if grep -q 'dataset_input_features.pop("observation.gripper_torque"' \
    "$LEROBOT_ROOT/src/lerobot/policies/factory.py"; then
    echo "Detected existing baseline torque-feature filter."
  else
    apply_once "$LEROBOT_ROOT" "$HANDOFF_ROOT/patches/lerobot_factory_refresh_dataset_features.patch"
  fi
  if grep -q '_get_padded_gripper_torque_window' "$ISAACLAB_ROOT/scripts/eval_server.py"; then
    echo "Detected existing IsaacLab torque-window server changes."
  else
    apply_once "$ISAACLAB_ROOT" "$HANDOFF_ROOT/patches/remote_workspace_isaaclab.patch"
  fi
fi

mkdir -p "$(dirname "$TEST_DST")" "$WORK_ROOT/experiment"
cp "$HANDOFF_ROOT/remote_handoff_gripper_lstm/lerobot_overrides/configuration_smolvla.py" "$CONFIG_DST"
cp "$HANDOFF_ROOT/remote_handoff_gripper_lstm/lerobot_overrides/modeling_smolvla.py" "$MODEL_DST"
cp "$HANDOFF_ROOT/remote_handoff_gripper_lstm/lerobot_overrides/test_smolvla_torque_lstm.py" "$TEST_DST"
cp -a "$HANDOFF_ROOT/remote_workspace/experiment/." "$WORK_ROOT/experiment/"
touch "$STAMP"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  ROOT="$WORK_ROOT" bash "$WORK_ROOT/experiment/setup_remote.sh"
else
  echo "Skipping dependency installation; set INSTALL_DEPS=1 to run setup_remote.sh."
fi

echo "PASS: remote workspace restored at $WORK_ROOT"
echo "Next: RUN_ISAAC_SMOKE=1 bash $WORK_ROOT/experiment/preflight_collection.sh"
