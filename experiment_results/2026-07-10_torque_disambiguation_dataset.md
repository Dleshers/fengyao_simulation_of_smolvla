# 2026-07-10 Torque-disambiguation diagnostic dataset

## Objective

Construct a dataset that truly requires gripper-torque information to choose the correct action.  The previous peg/preinsert MVP dataset did not satisfy this requirement: real torque, zero torque, and shuffled torque all reached nearly identical short-run training loss, indicating that the labels were solvable from vision/state alone or did not encode a torque-dependent decision.

This new dataset is deliberately a **diagnostic causal dataset**, not yet a physical peg-insertion benchmark.  It tests whether the gated/zero-init torque-LSTM injection path can use torque when the action label is ambiguous without it.

## Construction

Source raw HDF5:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_preinsert_rgb_20260709_5demo/peg_insert_demos.hdf5
```

Builder:

```text
experiment/build_torque_disambiguation_hdf5.py
```

The builder clones local trajectory windows from the source demos.  For each sampled source anchor, it creates three contact-mode episodes with the same state and RGB frames but different torque and action labels:

| Mode | Torque | Expert action |
| --- | ---: | --- |
| `contact_ok_proceed` | about `-25` | close gripper, proceed gently downward |
| `low_torque_slip_regrasp` | about `-1` | close gripper, lift slightly to recover/regrasp |
| `high_torque_jam_retreat` | about `-80` | open gripper, retreat upward |

Because the visual/state stream is copied while the correct action differs by torque mode, a visual-only policy cannot fully disambiguate the label.  Correctly aligned torque is the intended resolving variable.

Raw diagnostic HDF5:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_torque_disambiguation_20260710_45demo/peg_insert_demos.hdf5
```

Raw audit:

```text
OK: demos=45 total_steps=1800
OK: state=[49] action=[7] gripper_torque=[1]
OK: torque range=[-87.3327, -0.560193]
OK: rgb_table image_shapes=[(84, 84, 3)]
OK: rgb_wrist image_shapes=[(84, 84, 3)]
```

## LeRobot datasets

Converted with `--state-mode compact21` and `--drop-terminal-frame`.

| Dataset | Torque control | Episodes | Frames | Interface |
| --- | --- | ---: | ---: | --- |
| `Dleshers/peg-insert-torque-disambiguation-compact21-v1` | original | 45 | 1755 | state `[21]`, action `[7]`, two RGB `[3,224,224]`, torque `[30,1]` |
| `Dleshers/peg-insert-torque-disambiguation-compact21-zero-v1` | zero | 45 | 1755 | same |
| `Dleshers/peg-insert-torque-disambiguation-compact21-shuffleglobal-v1` | global shuffle | 45 | 1755 | same |

Uploaded Hugging Face revisions:

| Dataset | Revision |
| --- | --- |
| `Dleshers/peg-insert-torque-disambiguation-compact21-v1` | `dcfe4b38ceb6937f5cd13fa1c6f0da10450242a6` |
| `Dleshers/peg-insert-torque-disambiguation-compact21-zero-v1` | `1d2212d6366932b62b4460ccbd6537fd52a792bf` |
| `Dleshers/peg-insert-torque-disambiguation-compact21-shuffleglobal-v1` | `b7e83caef5ec0d6a3eb6926ffeffae441202e0db` |

All three Hub repositories are private as of upload.

Important converter update:

```text
experiment/convert_peg_insert_hdf5_to_lerobot.py
```

now supports:

```text
--torque-control shuffle_global
```

This matters because episode-level shuffle is insufficient when each diagnostic episode has nearly constant torque; it would preserve the contact-mode label.

## 300-step training validation

All runs used the official SmolVLA base checkpoint and compact21 state compatibility.  Torque runs used:

```text
input=1, hidden=32, layers=1, output=16, frozen
torque_zero_init_adapter=true
torque_gate_init=1.0
```

Training script:

```text
experiment/train_peg_insert_torque_mvp_smoke.sh
```

Outputs:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/
```

| Arm | Dataset | Final loss @300 | Last-10 mean loss |
| --- | --- | ---: | ---: |
| visual | original torque dataset, torque disabled | 0.043 | 0.0472 |
| real torque | original torque dataset, torque enabled | 0.008 | 0.0115 |
| zero torque | zero-torque control, torque enabled | 0.036 | 0.0405 |
| global-shuffle torque | globally shuffled torque control, torque enabled | 0.036 | 0.0404 |

Interpretation:

- Correct torque is strongly beneficial on this constructed diagnostic dataset.
- Zero torque and globally shuffled torque are almost identical, so the improvement is not merely caused by adding extra model parameters.
- The real-torque run is about `0.0115 / 0.0405 = 28%` of the zero-torque last-10 loss.
- Compared with visual-only, real torque is about `0.0115 / 0.0472 = 24%` of the last-10 loss.

## Checkpoint save behavior

All 300-step runs saved checkpoints successfully:

```text
checkpoints/000300/pretrained_model/model.safetensors
checkpoints/000300/pretrained_model/config.json
checkpoints/000300/pretrained_model/train_config.json
```

Torque-enabled runs still trigger the known safetensors shared/view-storage warning for:

```text
model.torque_lstm.lstm.weight_ih_l0
```

but the current patched save path falls back to cloned contiguous tensors and completes successfully:

```text
safetensors.save_model failed because of shared/view storage; saving a cloned contiguous state_dict instead
```

## Conclusion

This dataset successfully creates the missing diagnostic condition: the action label is intentionally ambiguous from vision/state and is resolved by torque.  The gated/zero-init torque injection does not collapse the official-pretrained SmolVLA policy in this setting; when torque is correctly aligned, it produces a clear short-run training advantage over visual-only, zero-torque, and shuffled-torque controls.

The result should be used as an architecture/data-path validation, not as a final task-performance claim.  A physical benchmark still needs real contact dynamics, preferably with slip/jam/regrasp outcomes generated by simulation physics or human/scripted teleoperation rather than synthetic labels.

## Recommended next command

Run a longer diagnostic training scale-up on the same dataset before investing in expensive physical collection:

```bash
ARM=torque \
DATASET_REPO_ID=Dleshers/peg-insert-torque-disambiguation-compact21-v1 \
STEPS=2000 SAVE_FREQ=1000 BATCH_SIZE=4 \
RUN_NAME=peg_insert_torque_disamb_torque_steps2000_seed1000_20260710 \
bash experiment/train_peg_insert_torque_mvp_smoke.sh
```

Then repeat matched 2000-step controls for `visual`, `zero`, and `shuffle`.
