#!/usr/bin/env bash
# Resume strict data collection, then convert, audit, and train controlled arms.
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
ISAAC="$RUNTIME_ROOT/IsaacLab-Tactile"
LEROBOT_ENV="$RUNTIME_ROOT/.venv/lerobot"
RUN="factory_peg_insert_rgb_proprio12_torque_formal_200demos_20260721_r1"
RAW="$RUNTIME_ROOT/persistent/raw_hdf5/$RUN/peg_insert_demos.hdf5"
DATA_ROOT="$RUNTIME_ROOT/persistent/lerobot_datasets"
OUT_ROOT="$RUNTIME_ROOT/persistent/gripper_lstm_experiments"
log(){ echo "[$(date -Is)] [FORMAL_PIPELINE] $*"; }
trap 's=$?; log "FAILED phase=${PHASE:-unknown} status=$s line=$LINENO"; exit $s' ERR
mkdir -p "$DATA_ROOT" "$OUT_ROOT"
PHASE=collect
log "collection resume begins raw=$RAW"
source "$RUNTIME_ROOT/.conda/isaaclab/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
"$ISAAC/isaaclab.sh" -p "$REPO_ROOT/experiment/collect_factory_peg_insert_formal.py" --headless --enable_cameras --experience "$ISAAC/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit" --dataset_file "$RAW" --num_demos 200 --max_attempts 300
PHASE=raw_audit
"$LEROBOT_ENV/bin/python" - <<PY
import h5py
p='$RAW'
with h5py.File(p,'r') as f:
    demos=f['demos']; assert len(demos)==200, len(demos)
    assert all(bool(demos[n].attrs.get('strict_success',False)) for n in demos)
    print('[FORMAL_RAW_AUDIT] strict_demos=',len(demos),'frames=',sum(len(demos[n]['state']) for n in demos))
PY
PHASE=convert_audit
for mode in original zero shuffle_episode; do
  case "$mode" in
    original) repo_id="Dleshers/factory-peg-insert-strict-rgb-proprio12-torque-v1";;
    zero) repo_id="Dleshers/factory-peg-insert-strict-rgb-proprio12-zero-torque-v1";;
    shuffle_episode) repo_id="Dleshers/factory-peg-insert-strict-rgb-proprio12-shuffle-torque-v1";;
  esac
  target="$DATA_ROOT/$repo_id"
  if [[ ! -d "$target/meta" ]]; then
    log "converting mode=$mode repo=$repo_id"
    "$LEROBOT_ENV/bin/python" "$REPO_ROOT/experiment/convert_factory_peg_insert_hdf5_to_lerobot.py" --input "$RAW" --output-dir "$DATA_ROOT" --repo-id "$repo_id" --torque-control "$mode" --use-videos
  else
    log "conversion already present repo=$repo_id"
  fi
  "$LEROBOT_ENV/bin/python" "$REPO_ROOT/experiment/audit_lerobot_dataset_features.py" --repo-id "$repo_id" --root "$target" --state-dim 12 --action-dim 6 --max-frames 1000
 done
PHASE=train
for arm in visual torque zero shuffle; do
  case "$arm" in
    visual|torque) repo_id="Dleshers/factory-peg-insert-strict-rgb-proprio12-torque-v1";;
    zero) repo_id="Dleshers/factory-peg-insert-strict-rgb-proprio12-zero-torque-v1";;
    shuffle) repo_id="Dleshers/factory-peg-insert-strict-rgb-proprio12-shuffle-torque-v1";;
  esac
  run_name="factory_strict_${arm}_50k_seed1000_20260721_r1"
  out="$OUT_ROOT/$run_name"
  if [[ -f "$out/checkpoints/last/pretrained_model/model.safetensors" || -f "$out/checkpoints/last/model.safetensors" ]]; then
    log "training already complete arm=$arm"
    continue
  fi
  log "training begins arm=$arm repo=$repo_id"
  ARM="$arm" DATASET_REPO_ID="$repo_id" DATASET_ROOT="$DATA_ROOT/$repo_id" OUTPUT_ROOT="$OUT_ROOT" RUN_NAME="$run_name" STATE_DIM=12 ACTION_DIM=6 STEPS=50000 BATCH_SIZE=8 SAVE_FREQ=50000 LOG_FREQ=50 NUM_WORKERS=2 bash "$REPO_ROOT/experiment/train_peg_insert_torque_mvp_smoke.sh"
done
log "COMPLETE conversion_audit_and_all_training"
