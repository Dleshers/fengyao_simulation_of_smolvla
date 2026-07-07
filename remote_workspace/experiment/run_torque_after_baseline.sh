#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work"
PERSISTENT_ROOT="/cs/student/project_msc/2025/rai/fenzhang/simulation_storage"
BASELINE_LOG="$PERSISTENT_ROOT/logs/post_collection_baseline.log"
DATASET_REPO_ID="franka_pickplace_joint_visual_torque_w30_v1"
DATASET_ROOT="$PERSISTENT_ROOT/datasets/$DATASET_REPO_ID"
PRETRAINED_POLICY="$WORK_ROOT/pretrained/smolvla_base_official"
TORQUE_WEIGHTS="$PERSISTENT_ROOT/trained_lstm_weights/torque_16d_encoder.pt"
OUTPUT_DIR="$PERSISTENT_ROOT/gripper_lstm_experiments/torque_lstm_smolvla_50000_seed1000"
SMOKE_DIR="$PERSISTENT_ROOT/gripper_lstm_experiments/torque_lstm_preflight_seed1000"
LEROBOT_ENV="$WORK_ROOT/.venv/lerobot"
HF="/cs/student/msc/rai/2025/fenzhang/.local/bin/hf"
HF_REPO="Dleshers/smolvla-franka-pickplace-torque-lstm-50k-seed1000"
MODEL_CARD="$PERSISTENT_ROOT/remote_handoff_gripper_lstm_workspace/experiment/hf_torque_lstm_model_card.md"

export TMPDIR="$WORK_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HOME="/scratch0/fenzhang/cache/huggingface"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
mkdir -p "$TMPDIR"

echo "[$(date -Is)] Waiting for successful 50K visual baseline"
while ! grep -q 'PASS: 50000-step pure SmolVLA baseline completed' "$BASELINE_LOG" 2>/dev/null; do
  sleep 60
done

echo "[$(date -Is)] Baseline complete; validating controlled torque-LSTM experiment"
test -s "$PRETRAINED_POLICY/model.safetensors"
test -s "$PRETRAINED_POLICY/config.json"
test -s "$TORQUE_WEIGHTS"
test -d "$DATASET_ROOT"
test ! -e "$OUTPUT_DIR"
test ! -e "$SMOKE_DIR"

"$LEROBOT_ENV/bin/python" "$WORK_ROOT/experiment/validate_dataset.py" \
  --repo-id "$DATASET_REPO_ID" --root "$DATASET_ROOT" \
  --window-size 30 --samples 64 --sequence-checks 1024
"$LEROBOT_ENV/bin/python" \
  "$PERSISTENT_ROOT/remote_handoff_gripper_lstm_workspace/experiment/verify_frozen_torque_encoder.py" \
  "$TORQUE_WEIGHTS"
"$LEROBOT_ENV/bin/python" -m pytest -q \
  "$WORK_ROOT/lerobot-tactile/tests/policies/smolvla/test_smolvla_torque_lstm.py" \
  "$WORK_ROOT/experiment/test_validate_dataset_windows.py"

COMMON=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --policy.path="$PRETRAINED_POLICY"
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.use_tactile=false
  --policy.use_torque_lstm=true
  --policy.torque_window_key=observation.gripper_torque
  --policy.torque_window_size=30
  --policy.torque_input_dim=1
  --policy.torque_lstm_hidden_dim=32
  --policy.torque_lstm_output_dim=16
  --policy.torque_lstm_num_layers=1
  --policy.torque_lstm_weights_path="$TORQUE_WEIGHTS"
  --policy.train_torque_lstm=false
  --seed=1000
  --batch_size=8
  --wandb.enable=false
)

cd "$WORK_ROOT/lerobot-tactile"
echo "[$(date -Is)] Running 2-step torque-LSTM integration smoke"
"$LEROBOT_ENV/bin/lerobot-train" "${COMMON[@]}" \
  --steps=2 --log_freq=1 --save_checkpoint=false --output_dir="$SMOKE_DIR"

echo "[$(date -Is)] Starting matched torque-LSTM training: 50000 steps"
echo "Only delta from baseline: frozen [30,1]->16 torque encoder plus trainable Action Expert suffix projection"
"$LEROBOT_ENV/bin/lerobot-train" "${COMMON[@]}" \
  --steps=50000 --log_freq=100 --save_freq=5000 --output_dir="$OUTPUT_DIR"

FINAL_DIR="$OUTPUT_DIR/checkpoints/050000/pretrained_model"
STATE_FILE="$OUTPUT_DIR/checkpoints/050000/training_state/training_step.json"
test -s "$FINAL_DIR/model.safetensors"
test -s "$FINAL_DIR/config.json"
test -s "$FINAL_DIR/train_config.json"
grep -Eq '(^|[^0-9])50000([^0-9]|$)' "$STATE_FILE"
"$LEROBOT_ENV/bin/python" "$WORK_ROOT/experiment/verify_torque_checkpoint.py" "$FINAL_DIR/model.safetensors"

install -m 0644 "$MODEL_CARD" "$FINAL_DIR/README.md"
"$HF" repos create "$HF_REPO" --type model --private --exist-ok
"$HF" upload "$HF_REPO" "$FINAL_DIR" --repo-type model --private \
  --commit-message "Upload completed 50K frozen torque-LSTM SmolVLA"
"$HF" models info "$HF_REPO" --format json
echo "[$(date -Is)] PASS: torque-LSTM 50K training and private Hugging Face upload completed"
