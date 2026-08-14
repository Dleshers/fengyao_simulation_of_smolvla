#!/usr/bin/env bash
# Resume-safe collection -> audit -> merge -> LeRobot conversion -> HF upload.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
PY="$RUNTIME_ROOT/.conda/isaaclab/bin/python"
ISAAC="$RUNTIME_ROOT/IsaacLab-Tactile/isaaclab.sh"
EXP="$RUNTIME_ROOT/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit"
export CONDA_PREFIX="$RUNTIME_ROOT/.conda/isaaclab"
export LEROBOT_SOURCE="$RUNTIME_ROOT/lerobot-tactile/src"
export PYTHONPATH="$LEROBOT_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y TERM=xterm
export HF_HUB_ENABLE_HF_TRANSFER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

VISUAL_POLICY="$RUNTIME_ROOT/persistent/models/contact_recovery_reactive_phase5_firststepw5_gate_20260813/runs/contact_recovery_reactive_phase5_firststepw5_gate_20260813/arms/visual/checkpoints/005000/pretrained_model"
TORQUE_POLICY="$RUNTIME_ROOT/persistent/models/contact_recovery_reactive_phase5_firststepw5_gate_20260813/runs/contact_recovery_reactive_phase5_firststepw5_gate_20260813/arms/torque/checkpoints/005000/pretrained_model"
META_REPO_ID="Dleshers/factory-peg-insert-contact-recovery-v1-7d-reactive-phase5-local"
META_ROOT="$RUNTIME_ROOT/persistent/lerobot_datasets/$META_REPO_ID"
BALANCED_RAW="$RUNTIME_ROOT/persistent/raw_hdf5/contact_recovery_native_v1_balanced64_20260811/peg_insert_demos.hdf5"
HARD_ROOT="$RUNTIME_ROOT/persistent/raw_hdf5/policy_failure_recovery_v1_hard16_20260814"
HARD_RAW="$HARD_ROOT/peg_insert_demos.hdf5"
BUILD_ROOT="$RUNTIME_ROOT/persistent/dataset_builds/contact_recovery_v2_actual_failure80_20260814"
HARD_AUDIT="$BUILD_ROOT/audit_hard16"
BALANCED_AUDIT="$BUILD_ROOT/audit_balanced64"
COMBINED_RAW="$BUILD_ROOT/combined80/peg_insert_demos.hdf5"
MANIFEST="$BUILD_ROOT/combined80/manifest.json"
TRAIN_REPO_ID="Dleshers/factory-peg-insert-contact-recovery-v2-hard80-lerobot"
TRAIN_PARENT="$RUNTIME_ROOT/persistent/lerobot_datasets"
TRAIN_ROOT="$TRAIN_PARENT/$TRAIN_REPO_ID"
RAW_REPO_ID="Dleshers/factory-peg-insert-policy-failure-recovery-v1-hard16"
UPLOAD_ROOT="$BUILD_ROOT/hf_raw_upload"
HANDOFF="$REPO_ROOT/experiment/ACTUAL_SUBMM_FAILURE_A100_TRAINING_HANDOFF_20260814.md"
TARGET="${TARGET:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-160}"
SEED="${SEED:-20261814}"
MIN_FREE_GIB="${MIN_FREE_GIB:-4}"

log() { echo "[$(date -Is)] [SUBMM_PIPELINE] $*"; }
trap 'log "FAIL line=$LINENO status=$?"' ERR

space_guard() {
  local available
  available=$(df -Pk "$RUNTIME_ROOT" | awk 'NR==2 {print $4}')
  if (( available < MIN_FREE_GIB * 1024 * 1024 )); then
    log "insufficient disk: $((available / 1024 / 1024)) GiB free, need $MIN_FREE_GIB GiB"
    exit 3
  fi
  log "disk_free_gib=$((available / 1024 / 1024))"
}

upload_retry() {
  local repo_id="$1" local_path="$2"
  local attempt
  for attempt in 1 2 3 4 5; do
    log "HF upload repo=$repo_id attempt=$attempt"
    if hf upload-large-folder "$repo_id" "$local_path" --repo-type dataset --num-workers 3; then
      return 0
    fi
    sleep "$((attempt * 30))"
  done
  return 1
}

mkdir -p "$HARD_ROOT" "$BUILD_ROOT" "$TRAIN_PARENT"
space_guard
hf auth whoami
log "collect/resume target=$TARGET seed_start=$SEED"
"$ISAAC" -p "$REPO_ROOT/experiment/collect_factory_peg_insert_policy_failure_recovery_v1.py"   --headless --enable_cameras --experience "$EXP"   --dataset-file "$HARD_RAW"   --visual-policy-path "$VISUAL_POLICY"   --torque-policy-path "$TORQUE_POLICY"   --dataset-root "$META_ROOT" --repo-id "$META_REPO_ID"   --num-demos "$TARGET" --max-attempts "$MAX_ATTEMPTS"   --policy-steps 180 --inference-samples 3 --coarse-until-xy-m 0.0035   --seed "$SEED"

log "audit hard failures"
"$PY" "$REPO_ROOT/experiment/audit_factory_peg_insert_policy_failure_recovery_v1.py"   --input "$HARD_RAW" --output-dir "$HARD_AUDIT"   --expected-demos "$TARGET" --require-balanced-grid

log "audit legacy balanced64"
"$PY" "$REPO_ROOT/experiment/audit_factory_peg_insert_contact_recovery_native_v1.py"   --input "$BALANCED_RAW" --output-dir "$BALANCED_AUDIT" --expected-demos 64

space_guard
if [[ ! -e "$COMBINED_RAW" ]]; then
  log "materialize shared 64+16 corpus"
  "$PY" "$REPO_ROOT/experiment/materialize_factory_peg_insert_contact_recovery_v2.py"     --balanced-input "$BALANCED_RAW" --failure-input "$HARD_RAW"     --output "$COMBINED_RAW" --manifest "$MANIFEST"     --expected-balanced 64 --expected-failures "$TARGET"
else
  log "combined raw already exists; preserving it"
fi

space_guard
if [[ ! -e "$TRAIN_ROOT/meta/info.json" ]]; then
  log "convert shared corpus to LeRobot 30x7 signed-torque format"
  "$PY" "$REPO_ROOT/experiment/convert_factory_peg_insert_hdf5_to_lerobot.py"     --input "$COMBINED_RAW" --output-dir "$TRAIN_PARENT" --repo-id "$TRAIN_REPO_ID"     --torque-control original --torque-dim 7 --policy-label-only --policy-phase-min 5     --fps 15
else
  log "LeRobot dataset already finalized; preserving it"
fi

mkdir -p "$UPLOAD_ROOT/hard16" "$UPLOAD_ROOT/combined80" "$UPLOAD_ROOT/audits/hard16" "$UPLOAD_ROOT/audits/balanced64"
ln -f "$HARD_RAW" "$UPLOAD_ROOT/hard16/peg_insert_demos.hdf5"
ln -f "$COMBINED_RAW" "$UPLOAD_ROOT/combined80/peg_insert_demos.hdf5"
cp "$MANIFEST" "$UPLOAD_ROOT/combined80/manifest.json"
cp "$HARD_AUDIT/audit.json" "$HARD_AUDIT/QUALITY_AUDIT.md" "$UPLOAD_ROOT/audits/hard16/"
cp "$BALANCED_AUDIT/audit.json" "$BALANCED_AUDIT/QUALITY_AUDIT.md" "$UPLOAD_ROOT/audits/balanced64/"
cp "$HANDOFF" "$UPLOAD_ROOT/README.md"
cp "$HANDOFF" "$TRAIN_ROOT/README.md"

"$PY" - "$HARD_RAW" "$COMBINED_RAW" "$MANIFEST" "$UPLOAD_ROOT/completion.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

hard, combined, manifest, output = map(Path, sys.argv[1:])
payload = {
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "hard16_sha256": digest(hard),
    "combined80_sha256": digest(combined),
    "manifest_sha256": digest(manifest),
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY
cp "$UPLOAD_ROOT/completion.json" "$TRAIN_ROOT/completion.json"

log "create private HF dataset repositories"
hf repos create "$RAW_REPO_ID" --repo-type dataset --private --exist-ok
hf repos create "$TRAIN_REPO_ID" --repo-type dataset --private --exist-ok
upload_retry "$RAW_REPO_ID" "$UPLOAD_ROOT"
upload_retry "$TRAIN_REPO_ID" "$TRAIN_ROOT"
log "COMPLETE raw=https://huggingface.co/datasets/$RAW_REPO_ID train=https://huggingface.co/datasets/$TRAIN_REPO_ID"
