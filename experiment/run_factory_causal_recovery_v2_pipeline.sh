#!/usr/bin/env bash
# Resume-safe sequential pipeline for causal, recovery-enriched Factory peg insertion.
# It deliberately stops on an incomplete/invalid raw file or a failed conversion.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
LEROBOT_ENV="$RUNTIME_ROOT/.venv/lerobot"
COLLECT_PID="${1:?collector launcher PID required}"
RUN="factory_peg_insert_causal_recovery_v2_formal120_20260724"
RAW="$RUNTIME_ROOT/persistent/raw_hdf5/$RUN/peg_insert_demos.hdf5"
DATA_ROOT="$RUNTIME_ROOT/persistent/lerobot_datasets"
OUT_ROOT="$RUNTIME_ROOT/persistent/gripper_lstm_experiments"
RUN_TAG="causal_recovery_v2_120demos_20260724_r1"

log() { echo "[$(date -Is)] [CAUSAL_V2_PIPELINE] $*"; }
trap 's=$?; log "FAILED phase=${PHASE:-unknown} status=$s line=$LINENO"; exit "$s"' ERR
mkdir -p "$DATA_ROOT" "$OUT_ROOT"

PHASE=wait_for_collection
log "waiting for collector pid=$COLLECT_PID"
while kill -0 "$COLLECT_PID" 2>/dev/null; do sleep 30; done

PHASE=raw_audit
"$LEROBOT_ENV/bin/python" - "$RAW" <<'PY'
import sys
import h5py
import numpy as np

with h5py.File(sys.argv[1], 'r') as f:
    assert f.attrs['format'] == 'factory_peg_insert_causal_recovery_v2'
    demos = f['demos']
    assert len(demos) == 120, len(demos)
    recovery = 0
    recovery_frames = 0
    frames = 0
    for name in demos:
        g = demos[name]
        assert bool(g.attrs['strict_success'])
        assert g.attrs['frame_alignment'] == 'pre_action'
        n = len(g['action'])
        assert n > 0 and g['state'].shape == (n, 12)
        assert g['joint_torque'].shape == (n, 7)
        assert g['rgb_table'].shape[0] == n and g['rgb_side'].shape[0] == n
        assert np.isfinite(g['state'][:]).all() and np.isfinite(g['action'][:]).all()
        recovery += bool(g.attrs['recovery_episode'])
        recovery_frames += int(g['is_recovery'][:].sum())
        frames += n
    assert recovery >= 40, recovery
    assert recovery_frames > 0, recovery_frames
    print(f'[CAUSAL_V2_RAW_AUDIT] demos={len(demos)} frames={frames} recovery_episodes={recovery} recovery_frames={recovery_frames}', flush=True)
PY

PHASE=convert_and_dataset_audit
for mode in original zero shuffle_episode; do
  case "$mode" in
    original) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-original" ;;
    zero) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-zero" ;;
    shuffle_episode) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-shuffle" ;;
  esac
  target="$DATA_ROOT/$repo_id"
  if [[ ! -e "$target" ]]; then
    log "converting mode=$mode repo=$repo_id"
    "$LEROBOT_ENV/bin/python" "$REPO_ROOT/experiment/convert_factory_peg_insert_hdf5_to_lerobot.py" \
      --input "$RAW" --output-dir "$DATA_ROOT" --repo-id "$repo_id" --torque-control "$mode" --use-videos
  else
    log "reusing existing conversion repo=$repo_id"
  fi
  "$LEROBOT_ENV/bin/python" "$REPO_ROOT/experiment/audit_lerobot_dataset_features.py" \
    --repo-id "$repo_id" --root "$target" --state-dim 12 --action-dim 6 --max-frames 0
done

PHASE=sequential_training
for arm in visual torque zero shuffle; do
  case "$arm" in
    visual|torque) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-original" ;;
    zero) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-zero" ;;
    shuffle) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-shuffle" ;;
  esac
  run_name="factory_${RUN_TAG}_${arm}_50k_seed1000"
  out="$OUT_ROOT/$run_name"
  if [[ -e "$out" ]]; then
    log "refusing to overwrite prior output arm=$arm path=$out"
    exit 2
  fi
  log "training begins arm=$arm repo=$repo_id"
  ARM="$arm" DATASET_REPO_ID="$repo_id" DATASET_ROOT="$DATA_ROOT/$repo_id" OUTPUT_ROOT="$OUT_ROOT" RUN_NAME="$run_name" \
    STATE_DIM=12 ACTION_DIM=6 STEPS=50000 BATCH_SIZE=8 SAVE_FREQ=50000 LOG_FREQ=50 NUM_WORKERS=2 \
    bash "$REPO_ROOT/experiment/train_peg_insert_torque_mvp_smoke.sh"
  log "training finished arm=$arm"
done

log "COMPLETE raw_audit_conversion_audit_and_sequential_training"
