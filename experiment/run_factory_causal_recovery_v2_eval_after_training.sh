#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
PIPELINE_PID="${1:?pipeline PID required}"
RUN_TAG="causal_recovery_v2_120demos_20260724_r1"
DATA_ROOT="$RUNTIME_ROOT/persistent/lerobot_datasets"
OUT_ROOT="$RUNTIME_ROOT/persistent/gripper_lstm_experiments"
ISAAC="$RUNTIME_ROOT/IsaacLab-Tactile"
EVAL_ROOT="$RUNTIME_ROOT/persistent/evaluation_results/$RUN_TAG"
log(){ echo "[$(date -Is)] [CAUSAL_V2_EVAL] $*"; }
trap 's=$?; log "FAILED status=$s line=$LINENO"; exit "$s"' ERR
while kill -0 "$PIPELINE_PID" 2>/dev/null; do sleep 60; done
mkdir -p "$EVAL_ROOT"
export LEROBOT_SOURCE="$RUNTIME_ROOT/lerobot-tactile/src"
for arm in visual torque zero shuffle; do
 case "$arm" in
  visual) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-original"; torque_mode=none;;
  torque) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-original"; torque_mode=original;;
  zero) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-zero"; torque_mode=zero;;
  shuffle) repo_id="Dleshers/factory-peg-insert-causal-recovery-v2-shuffle"; torque_mode=shuffle;;
 esac
 policy="$OUT_ROOT/factory_${RUN_TAG}_${arm}_50k_seed1000/checkpoints/050000/pretrained_model"
 [[ -f "$policy/model.safetensors" ]] || { log "missing completed checkpoint arm=$arm"; exit 2; }
 log "evaluation begins arm=$arm torque_mode=$torque_mode"
 "$ISAAC/isaaclab.sh" -p "$REPO_ROOT/experiment/eval_factory_peg_insert_formal.py" --headless --enable_cameras --experience "$ISAAC/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit" --policy-path "$policy" --dataset-root "$DATA_ROOT/$repo_id" --repo-id "$repo_id" --output "$EVAL_ROOT/${arm}_10ep_seed4100_n1.json" --episodes 10 --seed 4100 --max-steps 360 --torque-mode "$torque_mode" --n-action-steps 1
 log "evaluation finished arm=$arm"
done
log "COMPLETE"
