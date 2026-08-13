#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla
RUNTIME=$REPO/_runtime/remote_handoff_gripper_lstm_work
CKPT_ROOT=$RUNTIME/persistent/imported_checkpoints/contact_recovery_native_v1_balanced64_20260812/runs/contact_recovery_native_v1_balanced64_20260811/arms
DATA_ROOT=$RUNTIME/persistent/lerobot_datasets/Dleshers/factory-peg-insert-conditional-recovery-v3-7d-smoke4-local
REPO_ID=Dleshers/factory-peg-insert-conditional-recovery-v3-7d-smoke4-local
OUT=${OUT:-$RUNTIME/persistent/evaluation_results/native_contact_takeover_closed_loop_smoke_20260813}
EPISODES=${EPISODES:-4}
SEED=${SEED:-20261300}
mkdir -p "$OUT"

for arm in visual torque; do
  mode=none
  [[ "$arm" == torque ]] && mode=original
  echo "[CLOSED_LOOP] start arm=$arm episodes=$EPISODES seed=$SEED utc=$(date -u +%FT%TZ)"
  CONDA_PREFIX="$RUNTIME/.conda/isaaclab" OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y TERM=xterm \
    LEROBOT_SOURCE="$RUNTIME/lerobot-tactile/src" \
    "$RUNTIME/IsaacLab-Tactile/isaaclab.sh" -p "$REPO/experiment/eval_factory_peg_insert_native_contact_takeover.py" \
    --policy-path "$CKPT_ROOT/$arm/pretrained_model" --dataset-root "$DATA_ROOT" --repo-id "$REPO_ID" \
    --output "$OUT/$arm.json" --episodes "$EPISODES" --seed "$SEED" --max-steps 240 \
    --n-action-steps 1 --torque-mode "$mode" --headless --enable_cameras \
    --experience "$RUNTIME/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit" \
    > "$OUT/$arm.log" 2>&1
  echo "[CLOSED_LOOP] complete arm=$arm utc=$(date -u +%FT%TZ)"
done

"$RUNTIME/.conda/isaaclab/bin/python" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for arm in ('visual','torque'):
    s=json.loads((root/f'{arm}.json').read_text())
    rows.append((arm,s['episodes'],s['alignment_recoveries'],s['strict_recoveries'],s['n_action_steps']))
lines=['# Native-contact closed-loop smoke','',
       'Matched unseen seeds; native reset; physical rim contact; 30-frame 7D torque history; one action per observation.','',
       '| arm | episodes | aligned | strict | n_action_steps |','|---|---:|---:|---:|---:|']
lines += [f'| {a} | {e} | {x} | {y} | {n} |' for a,e,x,y,n in rows]
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(root/'REPORT.md')
PY
echo "[CLOSED_LOOP] complete all utc=$(date -u +%FT%TZ)"
