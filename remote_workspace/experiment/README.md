# Visual vs Gripper-Torque LSTM Experiment

## Layout

- Runtime workspace: `/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work`
- Persistent outputs: `/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/gripper_lstm_experiments`
- Isaac environment: `.conda/isaaclab`
- LeRobot environment: `.venv/lerobot`

## Pre-flight

The legacy `visual_050000*` and `torque_lstm_030000` checkpoints are incompatible
with the controlled interface and must not be used for initialization or reporting.

Required inputs:

1. An audited LeRobot dataset satisfying `SMOLVLA_TORQUE_INTERFACE.md`.
2. The original official `lerobot/smolvla_base` checkpoint (not the local
   `pretrained/smolvla_base` notebook-only directory).
3. A standalone causal LSTM checkpoint matching `[B,30,1] -> [B,16]`.
4. A GPU session where `nvidia-smi` succeeds.

Run source, GPU, storage and optional Isaac launch checks before collection:

```bash
RUN_ISAAC_SMOKE=1 bash experiment/preflight_collection.sh
```

```bash
.venv/lerobot/bin/python experiment/validate_dataset.py \
  --repo-id "$DATASET_REPO_ID" --root "$DATASET_ROOT"

.venv/lerobot/bin/python -m pytest -q \
  lerobot-tactile/tests/policies/smolvla/test_smolvla_torque_lstm.py \
  experiment/test_validate_dataset_windows.py
```

The torque sample must be float32 with trailing shape `[30,1]`. Index `-1`
is newest; episode starts are left-padded by repeating the first valid value.

## Matched training

Set all four input variables, then run:

```bash
export DATASET_REPO_ID=...
export DATASET_ROOT=...
export PRETRAINED_POLICY=lerobot/smolvla_base
export TORQUE_LSTM_WEIGHTS=/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/trained_lstm_weights/torque_16d_encoder.pt

STEPS=300 RUN_TAG=smoke experiment/train_pair.sh
STEPS=20000 RUN_TAG=full experiment/train_pair.sh
```

Use the same seed, split, checkpoint, cameras, optimizer, batch size and step
count. The only main-experiment switch is `use_torque_lstm`.

## Closed-loop evaluation

Terminal A:

```bash
experiment/run_eval_server.sh visual
# later restart as:
experiment/run_eval_server.sh torque
```

Terminal B:

```bash
VISUAL_POLICY=/path/to/visual TORQUE_POLICY=/path/to/torque experiment/eval_pair.sh
```

The evaluation scripts default to `CONTROL_MODE=joint`, matching the authoritative 9D-state/8D-action
interface. Do not override this with `ik_rel` for the controlled experiment.

Run at least seeds 1000, 1001 and 1002. Compare success, drops, collisions,
action differences, peak joint velocity/acceleration, safety violations and videos.
