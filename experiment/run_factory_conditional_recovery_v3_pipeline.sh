#!/usr/bin/env bash
# Detached, resume-safe formal data/fit/evaluation pipeline for the 7D torque study.
# It creates: 100 nominal insertions + 100 strict recovery successes per distance
# stratum.  Recovery episodes are duplicated once during conversion, so corrective
# frames have higher sampling mass without manipulating actions or observations.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
LEROBOT_ENV="${LEROBOT_ENV:-$RUNTIME_ROOT/.venv/lerobot}"
RUN="${RUN:-factory_peg_insert_conditional_recovery_v3_formal400_20260730}"
RUN_TAG="${RUN_TAG:-conditional_recovery_v3_400demos_7d_20260730_r1}"
RAW="$RUNTIME_ROOT/persistent/raw_hdf5/$RUN/peg_insert_demos.hdf5"
DATA_ROOT="$RUNTIME_ROOT/persistent/lerobot_datasets"
OUT_ROOT="$RUNTIME_ROOT/persistent/gripper_lstm_experiments"
EVAL_ROOT="$RUNTIME_ROOT/persistent/evaluation_results/$RUN_TAG"
STEPS="${STEPS:-50000}"
MIN_FREE_GIB="${MIN_FREE_GIB:-8}"

log() { echo "[$(date -Is)] [CONDITIONAL_V3] $*"; }
trap 's=$?; log "FAILED phase=${PHASE:-unknown} status=$s line=$LINENO"; exit "$s"' ERR
mkdir -p "$DATA_ROOT" "$OUT_ROOT" "$EVAL_ROOT" "$RUNTIME_ROOT/persistent/logs"

space_guard() {
  local avail_kib
  avail_kib=$(df -Pk "$RUNTIME_ROOT" | awk 'NR==2 {print $4}')
  if (( avail_kib < MIN_FREE_GIB * 1024 * 1024 )); then
    log "refusing phase=$PHASE: only $((avail_kib/1024/1024))GiB free (<${MIN_FREE_GIB}GiB)"
    exit 3
  fi
  log "space audit: $((avail_kib/1024/1024))GiB free"
}

PHASE=collect
space_guard
if [[ "${SKIP_COLLECTION:-0}" == "1" ]]; then
  [[ -f "$RAW" ]] || { log "SKIP_COLLECTION=1 but raw file is absent: $RAW"; exit 2; }
  log "SKIP_COLLECTION=1; using transferred raw=$RAW"
else
  log "collecting/resuming raw=$RAW"
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y TERM=xterm "$RUNTIME_ROOT/IsaacLab-Tactile/isaaclab.sh" -p \
    "$REPO_ROOT/experiment/collect_factory_peg_insert_conditional_recovery.py" \
    --dataset-file "$RAW" --normal-demos 100 --per-stratum 100 --max-attempts 2200 \
    --normal-max-steps 480 --recovery-max-steps 220 --resolution 84 --seed 20260730 \
    --headless --enable_cameras \
    --experience "$RUNTIME_ROOT/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit"
fi

PHASE=raw_audit
space_guard
"$LEROBOT_ENV/bin/python" - "$RAW" <<'PY'
import sys
import h5py
import numpy as np

path = sys.argv[1]
with h5py.File(path, 'r') as f:
    assert f.attrs['format'] == 'factory_peg_insert_conditional_recovery_v3'
    demos = f['demos']
    counts = {'normal': 0, 'easy': 0, 'medium': 0, 'hard': 0}
    frames = recovery_frames = 0
    for name in demos:
        g = demos[name]
        kind = g.attrs['difficulty_stratum']
        if isinstance(kind, bytes): kind = kind.decode()
        assert kind in counts, kind
        counts[kind] += 1
        n = len(g['action'])
        assert n > 0 and bool(g.attrs['strict_success'])
        assert g.attrs['frame_alignment'] == 'pre_action'
        assert g['state'].shape == (n, 12) and g['joint_torque'].shape == (n, 7)
        assert g['rgb_table'].shape[0] == n and g['rgb_side'].shape[0] == n
        assert np.isfinite(g['state'][:]).all() and np.isfinite(g['joint_torque'][:]).all()
        is_recovery = np.asarray(g['is_recovery'][:], dtype=bool)
        assert bool(is_recovery.all()) == (kind != 'normal')
        if kind != 'normal':
            xy, z = float(g.attrs['initial_xy_error_m']), float(g.attrs['initial_depth_m'])
            assert xy >= 0.0025 and 0.001 < z <= 0.008, (name, xy, z)
        frames += n
        recovery_frames += int(is_recovery.sum())
    assert counts == {'normal': 100, 'easy': 100, 'medium': 100, 'hard': 100}, counts
    assert recovery_frames > frames // 2, (recovery_frames, frames)
    print('[CONDITIONAL_V3_RAW_AUDIT]', {'demos': len(demos), 'counts': counts, 'frames': frames, 'recovery_frames': recovery_frames}, flush=True)
PY

if [[ "${STOP_AFTER_RAW:-0}" == "1" ]]; then
  log "STOP_AFTER_RAW=1; raw collection/audit complete, not converting or training locally"
  exit 0
fi

PHASE=convert_and_dataset_audit
for mode in original zero shuffle_episode; do
  case "$mode" in
    original) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-original" ;;
    zero) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-zero" ;;
    shuffle_episode) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-shuffle" ;;
  esac
  target="$DATA_ROOT/$repo_id"
  if [[ -e "$target" ]]; then
    log "reusing existing converted dataset mode=$mode root=$target"
  else
    space_guard
    log "converting mode=$mode with 7D torque and recovery-repeat=2"
    "$LEROBOT_ENV/bin/python" "$REPO_ROOT/experiment/convert_factory_peg_insert_hdf5_to_lerobot.py" \
      --input "$RAW" --output-dir "$DATA_ROOT" --repo-id "$repo_id" --torque-control "$mode" \
      --torque-dim 7 --recovery-repeat 2 --use-videos
  fi
  "$LEROBOT_ENV/bin/python" "$REPO_ROOT/experiment/audit_lerobot_dataset_features.py" \
    --repo-id "$repo_id" --root "$target" --state-dim 12 --action-dim 6 --torque-dim 7 --max-frames 0
done

PHASE=sequential_training
for arm in visual torque zero shuffle; do
  case "$arm" in
    visual|torque) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-original" ;;
    zero) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-zero" ;;
    shuffle) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-shuffle" ;;
  esac
  run_name="factory_${RUN_TAG}_${arm}_${STEPS}_seed1000"
  out="$OUT_ROOT/$run_name"
  if [[ -e "$out" ]]; then
    log "existing training directory requires manual inspection: $out"
    exit 4
  fi
  space_guard
  log "training starts arm=$arm run=$run_name"
  ARM="$arm" DATASET_REPO_ID="$repo_id" DATASET_ROOT="$DATA_ROOT/$repo_id" OUTPUT_ROOT="$OUT_ROOT" RUN_NAME="$run_name" \
    STATE_DIM=12 ACTION_DIM=6 TORQUE_INPUT_DIM=7 TRAIN_TORQUE_LSTM=true TORQUE_LSTM_WEIGHTS='' \
    STEPS="$STEPS" BATCH_SIZE=8 SAVE_FREQ="$STEPS" LOG_FREQ=50 NUM_WORKERS=2 \
    bash "$REPO_ROOT/experiment/train_peg_insert_torque_mvp_smoke.sh"
  log "training finished arm=$arm"
done

if [[ "${SKIP_EVALUATION:-0}" == "1" ]]; then
  log "SKIP_EVALUATION=1; four-arm training complete, no local Isaac Sim evaluation requested"
  exit 0
fi

PHASE=matched_evaluation
space_guard
for arm in visual torque zero shuffle; do
  case "$arm" in
    visual|torque) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-original" ;;
    zero) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-zero" ;;
    shuffle) repo_id="Dleshers/factory-peg-insert-conditional-recovery-v3-7d-shuffle" ;;
  esac
  case "$arm" in visual) torque_mode=none ;; torque) torque_mode=original ;; zero) torque_mode=zero ;; shuffle) torque_mode=shuffle ;; esac
  policy="$OUT_ROOT/factory_${RUN_TAG}_${arm}_${STEPS}_seed1000/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
  [[ -d "$policy" ]] || { log "missing policy=$policy"; exit 5; }
  for spec in 'easy 34 0.0038 0.0045' 'medium 33 0.0048 0.0059' 'hard 33 0.0061 0.0070'; do
    read -r stratum episodes pmin pmax <<<"$spec"
    log "evaluating arm=$arm stratum=$stratum episodes=$episodes"
    LEROBOT_SOURCE="$RUNTIME_ROOT/lerobot-tactile/src" OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y TERM=xterm \
      "$RUNTIME_ROOT/IsaacLab-Tactile/isaaclab.sh" -p "$REPO_ROOT/experiment/eval_factory_peg_insert_deterministic_rim_recovery.py" \
      --policy-path "$policy" --dataset-root "$DATA_ROOT/$repo_id" --repo-id "$repo_id" \
      --output "$EVAL_ROOT/${arm}_${stratum}.json" --episodes "$episodes" --seed 5800 --max-steps 240 \
      --torque-mode "$torque_mode" --perturb-min "$pmin" --perturb-max "$pmax" --initial-depth 0.003 \
      --headless --enable_cameras --experience "$RUNTIME_ROOT/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit"
  done
done

PHASE=report
"$LEROBOT_ENV/bin/python" - "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob('*.json')):
    s = json.loads(path.read_text())
    rows.append((path.stem, s['valid_initializations'], s['alignment_recovery_rate'], s['strict_recovery_rate']))
lines = ['# Conditional recovery v3: matched 100-episode evaluation', '', '| arm/stratum | valid starts | alignment rate | strict recovery rate |', '|---|---:|---:|---:|']
lines += [f'| {n} | {v} | {a:.3f} | {r:.3f} |' for n,v,a,r in rows]
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('[CONDITIONAL_V3_REPORT]', root/'REPORT.md', flush=True)
PY

if [[ "${STOP_AFTER_RAW:-0}" == "1" ]]; then
  log "STOP_AFTER_RAW=1; raw collection/audit complete, not converting or training locally"
  exit 0
fi
log "COMPLETE collection_conversion_four_arm_training_and_matched_evaluation"
