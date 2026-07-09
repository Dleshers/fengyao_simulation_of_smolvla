#!/usr/bin/env bash
set -euo pipefail

# Evaluate the gated/zero-init torque-LSTM continuation on the same seed set
# used by visual50k, visual-only 5k, and ungated torque 5k.
#
# Terminal A:
#   PORT=5562 bash experiment/eval_gated_torque_n10.sh server
#
# Terminal B:
#   PORT=5562 bash experiment/eval_gated_torque_n10.sh client

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PERSISTENT_ROOT="${PERSISTENT_ROOT:-$ROOT/_runtime/remote_handoff_gripper_lstm_work/persistent}"
RUN_NAME="${RUN_NAME:-targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_seed1000_20260709}"
EVAL_NAME="${EVAL_NAME:-targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_eval_n10_seed1000_20260709}"
POLICY_PATH="${POLICY_PATH:-$PERSISTENT_ROOT/gripper_lstm_experiments/$RUN_NAME/checkpoints/005000/pretrained_model}"
OUT_DIR="${OUT_DIR:-$PERSISTENT_ROOT/eval_diagnostics/$EVAL_NAME}"
PORT="${PORT:-5562}"
MODE="${1:-client}"

mkdir -p "$OUT_DIR" "$PERSISTENT_ROOT/logs"

case "$MODE" in
  server)
    export PORT
    export CONTROL_MODE="${CONTROL_MODE:-joint}"
    export TRAJECTORY_LOG="${TRAJECTORY_LOG:-$OUT_DIR/trajectory.jsonl}"
    bash _runtime/remote_handoff_gripper_lstm_work/experiment/run_eval_server.sh torque 2>&1 \
      | tee "$PERSISTENT_ROOT/logs/${EVAL_NAME}_server.log"
    ;;
  client)
    _runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/lerobot-eval \
      --policy.path="$POLICY_PATH" \
      --env.type=isaaclab_tactile_remote \
      --env.server_host=localhost \
      --env.server_port="$PORT" \
      --env.task=pick_place \
      --env.observation_height=224 \
      --env.observation_width=224 \
      --env.control_mode=joint \
      --env.torque_window_size=30 \
      --env.include_gripper_torque_window=true \
      --rename_map='{"observation.images.rgb_table":"observation.images.camera1","observation.images.rgb_wrist":"observation.images.camera2"}' \
      --eval.n_episodes=10 \
      --eval.batch_size=1 \
      --seed=1000 \
      --output_dir="$OUT_DIR" 2>&1 \
      | tee "$PERSISTENT_ROOT/logs/${EVAL_NAME}_client.log"
    ;;
  *)
    echo "Usage: $0 [server|client]" >&2
    exit 2
    ;;
esac
