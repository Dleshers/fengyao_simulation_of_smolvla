#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work"
PERSISTENT_ROOT="/cs/student/project_msc/2025/rai/fenzhang/simulation_storage"
COLLECTION_PID="${1:?usage: $0 COLLECTION_PID}"
RAW_DIR="$PERSISTENT_ROOT/datasets/franka_pickplace_joint_torque_w30_20260706T130538Z"
DATASET_REPO_ID="franka_pickplace_joint_visual_torque_w30_v1"
DATASET_ROOT="$PERSISTENT_ROOT/datasets/$DATASET_REPO_ID"
PRETRAINED_POLICY="$WORK_ROOT/pretrained/smolvla_base_official"
OUTPUT_DIR="$PERSISTENT_ROOT/gripper_lstm_experiments/baseline_smolvla_50000_seed1000"
LEROBOT_ENV="$WORK_ROOT/.venv/lerobot"

export TMPDIR="$WORK_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HOME="$WORK_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
mkdir -p "$TMPDIR"

echo "[$(date -Is)] Waiting for collection PID $COLLECTION_PID"
while kill -0 "$COLLECTION_PID" 2>/dev/null; do
  completed="$(find "$RAW_DIR/metadata" -maxdepth 1 -type f -name 'demo_*.json' 2>/dev/null | wc -l)"
  echo "[$(date -Is)] Collection progress: $completed/200"
  sleep 60
done

echo "[$(date -Is)] Collector exited; requiring finalized and checksummed HDF5"
test -s "$RAW_DIR/data.hdf5"
test -s "$RAW_DIR/data.hdf5.sha256"
(cd "$RAW_DIR" && sha256sum -c data.hdf5.sha256)
"$LEROBOT_ENV/bin/python" "$WORK_ROOT/experiment/inspect_raw_hdf5.py" "$RAW_DIR/data.hdf5"

echo "[$(date -Is)] Converting audited HDF5 to LeRobot"
if [[ ! -d "$DATASET_ROOT" ]]; then
  RAW_DIR="$RAW_DIR" DATASET_REPO_ID="$DATASET_REPO_ID" \
    PERSISTENT_ROOT="$PERSISTENT_ROOT" RUN_ISAAC_SMOKE=0 \
    bash "$WORK_ROOT/experiment/rebuild_dataset.sh" convert
else
  echo "Dataset already exists; validating without overwriting: $DATASET_ROOT"
fi

echo "[$(date -Is)] Validating shared camera/state/action and torque-window interfaces"
"$LEROBOT_ENV/bin/python" "$WORK_ROOT/experiment/validate_dataset.py" \
  --repo-id "$DATASET_REPO_ID" --root "$DATASET_ROOT" \
  --window-size 30 --samples 64 --sequence-checks 1024
"$LEROBOT_ENV/bin/python" -m pytest -q \
  "$WORK_ROOT/lerobot-tactile/tests/policies/smolvla/test_smolvla_torque_lstm.py" \
  "$WORK_ROOT/experiment/test_validate_dataset_windows.py"

test -s "$PRETRAINED_POLICY/model.safetensors"
test -s "$PRETRAINED_POLICY/config.json"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to overwrite training output: $OUTPUT_DIR" >&2
  exit 2
fi

echo "[$(date -Is)] Starting pure SmolVLA baseline: 50000 steps"
echo "Interface: camera1/2=[3,224,224], state=[9], action=[8], torque excluded from baseline policy"
cd "$WORK_ROOT/lerobot-tactile"
"$LEROBOT_ENV/bin/lerobot-train" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$PRETRAINED_POLICY" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_torque_lstm=false \
  --seed=1000 \
  --batch_size=8 \
  --steps=50000 \
  --log_freq=100 \
  --save_freq=5000 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR"

echo "[$(date -Is)] PASS: 50000-step pure SmolVLA baseline completed"
