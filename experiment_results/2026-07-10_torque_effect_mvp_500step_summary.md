# 2026-07-10 Torque-effect MVP 500-step training summary

## Goal

Continue the minimal viable torque-effect experiment after the 10-step smoke
training passed. The immediate question was:

> Under official `lerobot/smolvla_base` initialization, can the compact
> contact-rich/preinsert dataset support a controlled comparison between
> visual-only, real torque, zero torque, and shuffled torque?

This run intentionally stayed small. It is a training/data validation step, not
a closed-loop performance claim.

## Data used

Raw HDF5:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_preinsert_rgb_20260709_5demo/peg_insert_demos.hdf5
```

Raw audit:

- 5 demos
- 910 raw steps
- raw state `[49]`
- action `[7]`
- two RGB cameras `(84,84,3)`
- raw gripper torque `[1]`
- torque range approximately `[-80, 9.57778]`

Official-base-compatible LeRobot datasets:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-v1
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-zero-torque-v1
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-franka-oracle-preinsert-rgb-5demo-compact21-shuffle-torque-v1
```

LeRobot interface:

- episodes: 5
- frames: 905
- `observation.state [21]`
- `action [7]`
- `observation.images.camera1 [3,224,224]`
- `observation.images.camera2 [3,224,224]`
- `observation.gripper_torque [30,1]`

The compact21 state was necessary because official SmolVLA base uses a
pretrained `state_proj` with `max_state_dim=32`. Full `[49]` state is not
checkpoint-compatible without changing `state_proj` shape.

## Training setup

All runs used:

- official pretrained policy:
  `_runtime/remote_handoff_gripper_lstm_work/pretrained/official_smolvla_base`
- steps: 500
- batch size: 2
- seed: 1000
- save frequency: 500
- `policy.max_state_dim=32`
- `wandb.enable=false`

Torque arms used:

- `use_torque_lstm=true`
- `torque_window_key=observation.gripper_torque`
- `torque_window_size=30`
- `torque_input_dim=1`
- `torque_lstm_hidden_dim=32`
- `torque_lstm_output_dim=16`
- `torque_lstm_num_layers=1`
- `torque_lstm_weights_path=trained_lstm_weights/torque_16d_encoder.pt`
- `train_torque_lstm=false`
- `torque_zero_init_adapter=true`
- `torque_gate_init=1.0`

## Runs completed

| arm | dataset torque | run directory |
|---|---|---|
| visual | ignored | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_preinsert_compact21_visual_mvp_steps500_seed1000_20260710` |
| torque | original torque | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_preinsert_compact21_torque_mvp_steps500_seed1000_20260710` |
| zero | all zeros | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_preinsert_compact21_zero_torque_mvp_steps500_seed1000_20260710` |
| shuffle | episode-shuffled torque | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_preinsert_compact21_shuffle_torque_mvp_steps500_seed1000_20260710` |

All four runs completed and wrote:

```text
checkpoints/000500/pretrained_model/model.safetensors
checkpoints/000500/pretrained_model/train_config.json
checkpoints/000500/training_state/training_step.json
```

Torque checkpoint saving still triggers the known safetensors shared-storage
warning for `model.torque_lstm.lstm.weight_ih_l0`, but the patched saver falls
back to cloned contiguous tensors and successfully writes the model.

## Loss summary

Loss logs were emitted every 10 steps.

| arm | first loss @10 | final loss @500 | mean loss all logs | mean loss last 10 logs | min loss |
|---|---:|---:|---:|---:|---:|
| visual | 0.276 | 0.109 | 0.09036 | 0.05540 | 0.039 |
| torque | 0.479 | 0.045 | 0.09194 | 0.05590 | 0.031 |
| zero | 0.479 | 0.045 | 0.09196 | 0.05590 | 0.031 |
| shuffle | 0.479 | 0.045 | 0.09190 | 0.05600 | 0.032 |

## Checkpoint torque-branch inspection

At step 500:

| arm | torque gate | adapter weight std | adapter weight absmax |
|---|---:|---:|---:|
| torque | 0.9959 | 0.001240 | 0.00650 |
| zero | 0.9925 | 0.000914 | 0.00334 |
| shuffle | 0.9928 | 0.000920 | 0.00358 |

The real-torque adapter grows slightly more than zero/shuffle, but this does
not translate into an offline loss difference on this small MVP dataset.

## Interpretation

The current result supports the following limited claims:

1. The official SmolVLA base can train on the compact21 peg/preinsert dataset.
2. The gated/zero-init torque injection path does not crash or destabilize
   training at 500 steps.
3. The checkpoint save fix for torque-LSTM shared storage works in practice.
4. On this 5-demo `preinsert_alignment` MVP dataset, real torque is not
   distinguishable from zero or shuffled torque by offline training loss.

It does **not** support a claim that torque improves the task.

The most likely reason is experimental design, not necessarily that torque is
useless:

- The task is preinsert alignment, not true contact-rich insertion.
- The dataset is tiny: 5 demos / 905 frames.
- The action labels are mostly explained by state geometry and oracle phase.
- The visual/state inputs already reveal the intended motion.
- The torque signal is present, but not necessary to predict the oracle action.

## Recommendation

Do not spend compute on larger training for this exact 5-demo preinsert dataset.
The next useful step is to create a dataset where torque is causally needed.

Recommended next data design:

1. Keep the official-base-compatible compact state.
2. Expand to at least 50-100 automatic demos.
3. Add visual ambiguity/contact variation:
   - different friction/mass with similar visual appearance;
   - partial gripper/object occlusion;
   - contact/slip or failed grasp cases;
   - insertion/contact phases where visual state is insufficient.
4. Keep the same controls:
   - visual-only;
   - real torque;
   - zero torque;
   - shuffled torque.
5. Only after offline loss/action diagnostics show separation should closed-loop
   evaluation be used to claim task-level improvement.

## Suggested next command

Before collecting more data, implement a new oracle mode that records both
successful and controlled contact/slip/failure segments, rather than only
preinsert successes. The current collector should be extended instead of
starting formal training:

```bash
sed -n '1,260p' experiment/record_peg_insert_oracle_demos.py
```

Then add a new collection mode, for example:

```text
--success_mode contact_disambiguation
--include_failure_segments
--randomize_friction
--randomize_mass
```

The aim is to make the torque channel predictive of different corrective
actions under visually similar observations.
