#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla
RUNTIME=$REPO/_runtime/remote_handoff_gripper_lstm_work
INPUT=$RUNTIME/persistent/raw_hdf5/contact_recovery_native_v1_balanced64_20260811/peg_insert_demos.hdf5
OUTPUT_ROOT=$RUNTIME/persistent/lerobot_datasets
REPO_ID=Dleshers/factory-peg-insert-contact-recovery-v1-7d-reactive-phase5-local

PYTHONPATH="$RUNTIME/lerobot-tactile/src" \
  "$RUNTIME/.conda/isaaclab/bin/python" "$REPO/experiment/convert_factory_peg_insert_hdf5_to_lerobot.py" \
  --input "$INPUT" \
  --output-dir "$OUTPUT_ROOT" \
  --repo-id "$REPO_ID" \
  --torque-control original \
  --torque-dim 7 \
  --policy-label-only \
  --policy-phase-min 5

TARGET=$OUTPUT_ROOT/$REPO_ID
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["total_episodes"]==64, x; print({k:x.get(k) for k in ("total_episodes","total_frames","fps")})' "$TARGET/meta/info.json"
echo "[REACTIVE_PHASE5] conversion and 64-episode audit complete"
