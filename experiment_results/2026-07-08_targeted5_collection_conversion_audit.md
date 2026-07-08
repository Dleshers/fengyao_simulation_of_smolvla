# 2026-07-08 Targeted 5-Pose Collection, Clean Conversion, and Merge Decision

## Scope

This run collected small expert batches for 5 representative baseline-failure cube poses, converted each raw HDF5 into an independent clean LeRobot dataset with `--drop-terminal-frame`, and audited the result before deciding whether the data should be merged for training.

No official dataset, checkpoint, or Hugging Face repository was overwritten. All HDF5/parquet outputs are under ignored `_runtime/` storage.

## Pose selection

The formal dynamic baseline evaluation used 10 reset poses and produced 0/10 success. The reset cube positions were:

```text
0: [0.3861, -0.0917, 0.022]
1: [0.4563, -0.1064, 0.022]
2: [0.4978, -0.1132, 0.022]
3: [0.4278, -0.1147, 0.022]
4: [0.4654, -0.0693, 0.022]
5: [0.4251, -0.1044, 0.022]
6: [0.3504, -0.1736, 0.022]
7: [0.4319, -0.1230, 0.022]
8: [0.4830, -0.1037, 0.022]
9: [0.4652, -0.1470, 0.022]
```

Selected targeted poses:

| pose name | fixed cube pose |
|---|---|
| `center_cluster` | `[0.425, -0.115, 0.022]` |
| `right_center` | `[0.498, -0.113, 0.022]` |
| `right_upper` | `[0.465, -0.069, 0.022]` |
| `right_lower` | `[0.465, -0.147, 0.022]` |
| `left_lower` | `[0.350, -0.174, 0.022]` |

These cover the central repeated failures, the right-side failures, upper/lower y edges, and the left-lower boundary case.

## Raw HDF5 collection

Each pose was collected independently with:

- `NUM_DEMOS=3`
- `NUM_ENVS=1`
- fixed background
- cameras enabled
- fixed cube pose override

Raw HDF5 outputs:

| pose | raw path | demos | raw frames | SHA256 |
|---|---|---:|---:|---|
| `center_cluster` | `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted5_n3_20260708_center_cluster/data.hdf5` | 3 | 616 | `fa067a51dc98053b1b9e899066a467be0a9617c91050952cdbbef827938c83c7` |
| `right_center` | `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted5_n3_20260708_right_center/data.hdf5` | 3 | 613 | `c11429dde8cb05919d396bfaae3b9c235a2bb672014001b640bf827d07ca0b63` |
| `right_upper` | `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted5_n3_20260708_right_upper/data.hdf5` | 3 | 612 | `54ea560fcb88c216128fe2de3184a8fbe1f8743e45da20f7127c819d41380b53` |
| `right_lower` | `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted5_n3_20260708_right_lower/data.hdf5` | 3 | 613 | `62043d27feec1657f2299faac1cd56fcf1c3524871ba975fe058a2600e7d1c76` |
| `left_lower` | `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted5_n3_20260708_left_lower/data.hdf5` | 3 | 623 | `111101ad713310e273560ad968ede8579be45501582c15f3151474db4bc3c253` |

All raw HDF5 files passed `inspect_raw_hdf5.py`:

- state schema: `joint_state_9d:[arm_joint_pos(7),gripper_qpos(2)]`
- action schema: `joint_action_8d:[arm_joint_pos_target_abs(7),gripper_cmd(1)]`
- fps: `20`
- all demos successful
- all first cube poses exactly matched the requested fixed pose

Isaac still raises the known `ReferenceError: weakly-referenced object no longer exists` during camera shutdown. It remained non-fatal because each HDF5 was finalized and passed inspection.

## Clean LeRobot conversion

Each raw HDF5 was converted with `--drop-terminal-frame` to avoid the previously identified terminal reset action-label contamination.

Clean independent datasets:

| pose | repo id / local directory | episodes | clean frames |
|---|---|---:|---:|
| `center_cluster` | `targeted5_center_cluster_w30_n3_clean_v1` | 3 | 613 |
| `right_center` | `targeted5_right_center_w30_n3_clean_v1` | 3 | 610 |
| `right_upper` | `targeted5_right_upper_w30_n3_clean_v1` | 3 | 609 |
| `right_lower` | `targeted5_right_lower_w30_n3_clean_v1` | 3 | 610 |
| `left_lower` | `targeted5_left_lower_w30_n3_clean_v1` | 3 | 620 |

Total clean targeted data:

- episodes: `15`
- frames: `3062`

Standard validation passed for every clean dataset:

- `observation.state`: `[9]`
- `observation.images.camera1`: `[3,224,224]`
- `observation.images.camera2`: `[3,224,224]`
- `action`: `[8]`
- `observation.gripper_torque`: `[30,1]`
- causal torque-window audit passed

## Semantic boundary audit

Raw terminal action labels again showed reset-sized jumps, validating the need for `--drop-terminal-frame`:

| pose | raw terminal L2 mean | raw pre-terminal L2 mean | clean terminal L2 mean | clean terminal L2 max |
|---|---:|---:|---:|---:|
| `center_cluster` | `0.7986` | `0.0475` | `0.0475` | `0.0501` |
| `right_center` | `0.8361` | `0.0495` | `0.0495` | `0.0537` |
| `right_upper` | `0.8851` | `0.0573` | `0.0573` | `0.0609` |
| `right_lower` | `0.8006` | `0.0542` | `0.0542` | `0.0598` |
| `left_lower` | `0.7947` | `0.0506` | `0.0506` | `0.0517` |

After dropping the final raw frame, terminal reset contamination was removed.

The `left_lower` clean dataset has a larger non-terminal action-state L2 maximum (`0.3174`) than the other poses (`~0.13-0.16`). Manual inspection shows this peak occurs around mid-trajectory frames 101-104 during a continuous larger corrective motion, not at the episode boundary. This is not the reset-label bug, but it should be watched during training.

## Detailed audit artifacts

Generated local audit files:

- `experiment_results/2026-07-08_targeted5_collection_audit/targeted5_summary.csv`
- `experiment_results/2026-07-08_targeted5_collection_audit/targeted5_per_episode.csv`
- `experiment_results/2026-07-08_targeted5_collection_audit/targeted5_summary.json`

## Merge/training decision

Decision: **yes, merge for a controlled ablation, but do not replace the official dataset and do not launch full training yet.**

Rationale:

1. The targeted data directly covers the current baseline's observed failure poses.
2. All 15 raw demonstrations are successful expert rollouts.
3. Clean conversion removes terminal reset action-label contamination.
4. The resulting targeted set is modest relative to the official dataset:
   - official dataset: `200` episodes, `41,276` frames
   - targeted add-on: `15` episodes, `3,062` frames
   - approximate frame increase: `7.4%`
5. This is large enough to test targeted augmentation, but small enough to avoid overwhelming the original distribution.

Recommended next step:

- Build a separate merged dataset candidate, for example:
  `franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean`
- Keep the original dataset unchanged.
- Before any 50k training, run a short controlled ablation:
  - either `5k` steps from scratch with identical baseline hyperparameters,
  - or a short fine-tune from the baseline checkpoint only if explicitly approved.

Do not train on the unclean conversions or raw terminal frames.
