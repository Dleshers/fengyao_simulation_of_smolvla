#!/usr/bin/env bash
# Wait for the causal-collection smoke test, audit the completed HDF5, then
# launch the resumable formal collection.  Intended to be started with nohup.
set -euo pipefail

REPO=/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla
ROOT="$REPO/_runtime/remote_handoff_gripper_lstm_work"
SMOKE_PID="${1:?smoke launcher PID required}"
SMOKE_H5="$ROOT/persistent/raw_hdf5/factory_peg_insert_causal_recovery_v2_smoke12_20260724/peg_insert_demos.hdf5"
FORMAL_DIR="$ROOT/persistent/raw_hdf5/factory_peg_insert_causal_recovery_v2_formal120_20260724"
FORMAL_H5="$FORMAL_DIR/peg_insert_demos.hdf5"

while kill -0 "$SMOKE_PID" 2>/dev/null; do
  sleep 30
done

"$ROOT/.conda/isaaclab/bin/python" - "$SMOKE_H5" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as f:
    assert f.attrs["format"] == "factory_peg_insert_causal_recovery_v2"
    demos = f["demos"]
    assert len(demos) == 12, len(demos)
    recovery = sum(bool(demos[k].attrs["recovery_episode"]) for k in demos)
    recovery_frames = sum(int(demos[k]["is_recovery"][:].sum()) for k in demos)
    assert recovery >= 3, recovery
    assert recovery_frames > 0, recovery_frames
    print(f"[SMOKE_AUDIT] pass demos={len(demos)} recovery_episodes={recovery} recovery_frames={recovery_frames}", flush=True)
PY

mkdir -p "$FORMAL_DIR"
exec "$ROOT/IsaacLab-Tactile/isaaclab.sh" -p "$REPO/experiment/collect_factory_peg_insert_causal_recovery.py" \
  --headless --enable_cameras \
  --experience "$ROOT/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit" \
  --dataset-file "$FORMAL_H5" \
  --num-demos 120 --max-attempts 320 --recovery-fraction .5 --seed 20260724
