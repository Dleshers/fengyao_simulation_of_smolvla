# 2026-07-10 Torque-effect MVP dataset and smoke-training report

## Goal

Advance the minimal viable plan for testing whether gripper-torque information
helps, or at least does not destabilize, a contact-rich task under the official
SmolVLA pretraining condition.

This is **not yet** a final physical peg-insert benchmark. It is an MVP
pipeline checkpoint:

1. automated scripted/oracle data,
2. contact/pre-insertion behavior with real robot actions,
3. original/zero/shuffled torque controls,
4. official `lerobot/smolvla_base` initialization,
5. short smoke training for all comparison arms.

## Dataset collection

Collected a 5-demo RGB raw HDF5 dataset with:

- procedural peg + procedural target block,
- robot-action oracle, not object teleport,
- success mode: `preinsert_alignment`,
- RGB cameras enabled,
- gripper torque recorded as mean finger torque.

Raw HDF5:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_preinsert_rgb_20260709_5demo/peg_insert_demos.hdf5
```

Raw audit:

- demos: 5
- total raw steps: 910
- state: `[49]`
- action: `[7]`
- `rgb_table`: `(84,84,3)`
- `rgb_wrist`: `(84,84,3)`
- `gripper_torque`: `[1]`
- torque range: approximately `[-80, 9.57778]`

## Important SmolVLA official-base compatibility finding

The first direct training attempt using `observation.state[49]` failed because
official `lerobot/smolvla_base` has a pretrained `state_proj` shaped for
`max_state_dim=32`.

Attempting `--policy.max_state_dim=64` also failed, because the checkpoint
contains:

```text
model.state_proj.weight: [960, 32]
```

while the modified model expected:

```text
model.state_proj.weight: [960, 64]
```

Therefore, the official-pretrain-compatible MVP should not feed the full 49D
Isaac policy state directly. The dataset was converted to `compact21`.

## compact21 state definition

The compact state keeps the task-relevant pieces while staying below
`max_state_dim=32`:

```text
joint_pos_rel(9)
eef_pos(3)
peg_pos(3)
hole_pos(3)
peg_to_hole_pos(3)
```

Total:

```text
observation.state [21]
```

This preserves proprioception and task geometry, while remaining compatible
with the official SmolVLA state projection.

## LeRobot datasets produced

All use:

- episodes: 5
- frames: 905 after `--drop-terminal-frame`
- `observation.state`: `[21]`
- `action`: `[7]`
- `observation.images.camera1`: `[3,224,224]`
- `observation.images.camera2`: `[3,224,224]`
- `observation.gripper_torque`: `[30,1]`

Datasets:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-v1
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-zero-torque-v1
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-shuffle-torque-v1
```

Torque audits:

| dataset | newest torque min/max | newest mean/std | window mean/std |
|---|---:|---:|---:|
| original | `[-80, 9.57778]` | `-5.58816 / 7.81393` | `-11.3517 / 21.701` |
| zero | `[0, 0]` | `0 / 0` | `0 / 0` |
| shuffle | `[-80, 9.57778]` | `-5.60591 / 7.81033` | `-5.81285 / 7.4356` |

The shuffle control preserves the marginal torque scale but breaks temporal
alignment within each episode before causal windowing.

## Code changes

- `experiment/convert_peg_insert_hdf5_to_lerobot.py`
  - added `--state-mode full49|compact21`;
  - added `--torque-control original|zero|shuffle_episode`.
- `experiment/audit_lerobot_dataset_features.py`
  - added local LeRobot shape/stat audit.
- `experiment/train_peg_insert_torque_mvp_smoke.sh`
  - added official-base smoke training for arms:
    - `visual`
    - `torque`
    - `zero`
    - `shuffle`

## Smoke training results

All smoke runs use:

- official pretrained policy:
  `_runtime/remote_handoff_gripper_lstm_work/pretrained/official_smolvla_base`
- steps: 10
- batch size: 2
- seed: 1000
- `max_state_dim=32`
- `action[7]`

Runs:

| arm | torque input | status | final logged loss |
|---|---|---:|---:|
| visual | ignored | pass | `0.295` |
| torque | original torque | pass | `0.479` |
| zero | all-zero torque | pass | `0.479` |
| shuffle | episode-shuffled torque | pass | `0.478` |

Checkpoint directories:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_preinsert_compact21_visual_mvp_smoke_steps10_seed1000_20260710/checkpoints/000010/pretrained_model
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_preinsert_compact21_torque_mvp_smoke_steps10_seed1000_20260710/checkpoints/000010/pretrained_model
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_preinsert_compact21_zero_torque_mvp_smoke_steps10_seed1000_20260710/checkpoints/000010/pretrained_model
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_preinsert_compact21_shuffle_torque_mvp_smoke_steps10_seed1000_20260710/checkpoints/000010/pretrained_model
```

The torque arms still trigger the known safetensors shared-storage warning for
`model.torque_lstm.lstm.weight_ih_l0`, but the patched save path falls back to a
cloned contiguous state dict and successfully writes `model.safetensors`.

## Interpretation

The MVP now supports a controlled training comparison under official SmolVLA
pretraining:

- visual-only baseline,
- visual + real torque,
- visual + zero torque,
- visual + shuffled torque.

At 10 steps this only proves compatibility and non-crashing behavior. It does
not prove a torque benefit. The useful next experimental scale is still small:
roughly 100-500 training steps on this 5-demo dataset to compare whether the
real-torque arm diverges from the zero/shuffle controls in loss/action
prediction behavior.

## Remaining caveats

1. The current task is `preinsert_alignment`, not full physical insertion.
2. The procedural target is a solid block, not an insertable hole.
3. Factory hole USD root pose is unstable and should not be used until repaired.
4. 5 demos are enough for pipeline proof, not for a publishable effect claim.
5. For a real effect claim, expand to at least 50-100 automatically generated
   episodes with visual ambiguity/contact variation and evaluate closed-loop.

## Recommended next command

Run a still-small but more informative 500-step comparison, starting with
visual and real torque:

```bash
ARM=visual STEPS=500 SAVE_FREQ=500 BATCH_SIZE=2 \
RUN_NAME=peg_insert_preinsert_compact21_visual_mvp_steps500_seed1000_20260710 \
bash experiment/train_peg_insert_torque_mvp_smoke.sh

ARM=torque STEPS=500 SAVE_FREQ=500 BATCH_SIZE=2 \
RUN_NAME=peg_insert_preinsert_compact21_torque_mvp_steps500_seed1000_20260710 \
bash experiment/train_peg_insert_torque_mvp_smoke.sh
```

If both pass, repeat with `ARM=zero` and `ARM=shuffle`, then compare offline
loss curves and action prediction diagnostics before any closed-loop claim.
