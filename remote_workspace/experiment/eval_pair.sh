#!/usr/bin/env bash
set -euo pipefail

: "${VISUAL_POLICY:?Set VISUAL_POLICY}"
: "${TORQUE_POLICY:?Set TORQUE_POLICY}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ENV="${LEROBOT_ENV:-$ROOT/.venv/lerobot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/gripper_lstm_experiments}"
EPISODES="${EPISODES:-10}"
SEED="${SEED:-1000}"
CONTROL_MODE="${CONTROL_MODE:-joint}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"

run_eval() {
  local policy="$1" name="$2" include_torque="$3"
  "$LEROBOT_ENV/bin/lerobot-eval" \
    --policy.path="$policy" \
    --env.type=isaaclab_tactile_remote \
    --env.server_host=localhost --env.server_port="${PORT:-5555}" \
    --env.task=pick_place --env.torque_window_size=30 \
    --env.observation_height=224 --env.observation_width=224 \
    --env.control_mode="$CONTROL_MODE" \
    --rename_map='{"observation.images.rgb_table":"observation.images.camera1","observation.images.rgb_wrist":"observation.images.camera2"}' \
    --env.include_gripper_torque_window="$include_torque" \
    --eval.n_episodes="$EPISODES" --eval.batch_size=1 \
    --seed="$SEED" --output_dir="$OUTPUT_ROOT/eval_${name}_seed${SEED}"
}

echo "Start run_eval_server.sh visual in another terminal, then press Enter."
read -r
run_eval "$VISUAL_POLICY" visual false
echo "Restart run_eval_server.sh torque, then press Enter."
read -r
run_eval "$TORQUE_POLICY" torque_lstm true
