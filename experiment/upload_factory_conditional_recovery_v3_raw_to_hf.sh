#!/usr/bin/env bash
# Wait for the formal collector, audit completeness, then upload its raw HDF5
# to the private Hugging Face Dataset repository. This script never uploads a
# partial collection.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
COLLECTOR_PID="${1:?collector PID required}"
HF_DATASET_ID="${HF_DATASET_ID:-Dleshers/factory-peg-insert-conditional-recovery-v3-raw}"
RUN="${RUN:-factory_peg_insert_conditional_recovery_v3_formal400_20260730}"
RAW_DIR="$RUNTIME_ROOT/persistent/raw_hdf5/$RUN"
RAW="$RAW_DIR/peg_insert_demos.hdf5"
CARD="$REPO_ROOT/dataset_cards/factory_peg_insert_conditional_recovery_v3_RAW_README.md"

log() { echo "[$(date -Is)] [CONDITIONAL_V3_HF_UPLOAD] $*"; }
trap 's=$?; log "FAILED status=$s line=$LINENO"; exit "$s"' ERR

log "waiting for collector pid=$COLLECTOR_PID"
while kill -0 "$COLLECTOR_PID" 2>/dev/null; do sleep 60; done

"$RUNTIME_ROOT/.venv/lerobot/bin/python" - "$RAW" <<'PY'
import sys
import h5py
path = sys.argv[1]
with h5py.File(path, 'r') as f:
    assert f.attrs['format'] == 'factory_peg_insert_conditional_recovery_v3'
    counts = {'normal': 0, 'easy': 0, 'medium': 0, 'hard': 0}
    for g in f['demos'].values():
        assert bool(g.attrs['strict_success'])
        key = g.attrs['difficulty_stratum']
        if isinstance(key, bytes): key = key.decode()
        counts[key] += 1
    assert counts == {'normal': 100, 'easy': 100, 'medium': 100, 'hard': 100}, counts
print('[CONDITIONAL_V3_HF_UPLOAD] raw audit passed', flush=True)
PY

hf repos create "$HF_DATASET_ID" --type dataset --private --exist-ok
hf upload-large-folder "$HF_DATASET_ID" "$RAW_DIR" --type dataset --num-workers 4
hf upload "$HF_DATASET_ID" "$CARD" README.md --type dataset --commit-message "Add conditional recovery v3 dataset card"
log "COMPLETE dataset=https://huggingface.co/datasets/$HF_DATASET_ID"
