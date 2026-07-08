# 2026-07-07 dataset nearest-neighbor audit for seed1002 eval states

This audit compares selected failed closed-loop eval states against the local
LeRobot training parquet dataset.  It is offline-only and does not train,
evaluate in batch, or modify any dataset/checkpoint.

## Question

Is the current baseline failure more likely caused by a bug in the demonstration
generation script, or by another mismatch?

## Short answer

Current evidence does **not** support a hard demonstration-generation bug such
as wrong action labels, wrong gripper sign, or wrong joint/IK action semantics.

The more likely issue is a distribution mismatch: the policy encounters
closed-loop eval observations whose visual wrist-camera appearance is very
different from nearest training frames, even when the robot joint state is
similar.  The demonstration pipeline may still be an upstream contributor if it
produced only narrow, successful, teacher-forced visual states, but the observed
failure is not best explained as corrupted demonstration actions.

## Script and outputs

Script:

- `remote_workspace/experiment/dataset_nearest_eval_state_audit.py`

Inputs:

- Dataset:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1`
- Eval trajectory:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/baseline_eef_trajectory.jsonl`

Outputs:

- `experiment_results/2026-07-07_dataset_nearest_eval_state_audit_seed1002/nearest_rows.csv`
- `experiment_results/2026-07-07_dataset_nearest_eval_state_audit_seed1002/nearest_summary.json`
- `experiment_results/2026-07-07_dataset_nearest_eval_state_audit_seed1002/nearest_camera_montage.jpg`

Method:

- For selected eval steps `55,57,59,60,61,62,63,64,65,66,70`,
  construct raw eval state as:
  `[joint_pos_after(7), gripper_qpos_after(2)]`.
- Compare against all `41,276` dataset `observation.state` rows.
- Use dataset `observation.state` std for z-normalized distance.
- Extract top-5 nearest training frames, their actions, and embedded camera
  image statistics.

## Key nearest-neighbor results

Top-1 nearest neighbors:

| Eval step | Eval gripper cmd | Eval EEF-cube | Top train ep/frame | Train gripper action | Train cam2 mean | Eval cam2 mean |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 55 | 0.96757 | 0.14110 m | ep197/frame64 | 1.0 | 0.43862 | 0.59599 |
| 57 | 0.97008 | 0.13319 m | ep158/frame67 | 1.0 | 0.43378 | 0.59615 |
| 59 | 0.94824 | 0.12713 m | ep158/frame69 | 1.0 | 0.43077 | 0.59607 |
| 60 | 0.93769 | 0.12466 m | ep158/frame70 | 1.0 | 0.42957 | 0.59609 |
| 61 | 0.90270 | 0.12202 m | ep158/frame71 | 1.0 | 0.42846 | 0.59610 |
| 62 | 0.90890 | 0.11939 m | ep158/frame72 | -1.0 | 0.42700 | 0.59607 |
| 63 | -0.98592 | 0.11725 m | ep49/frame72 | -1.0 | 0.43677 | n/a |
| 64 | -1.01344 | 0.11495 m | ep51/frame106 | -1.0 | 0.43124 | n/a |
| 65 | -0.99400 | 0.11516 m | ep51/frame106 | -1.0 | 0.43124 | 0.59614 |
| 66 | -1.00880 | 0.11622 m | ep51/frame107 | -1.0 | 0.43008 | n/a |
| 70 | -1.00510 | 0.12213 m | ep51/frame114 | -1.0 | 0.43607 | 0.59621 |

Notes:

- The nearest training states are quite close in normalized joint space before
  closure.  Top-1 normalized state distances around steps `55-64` are roughly
  `0.34-0.39`.
- Training nearest-neighbor gripper labels switch from open to close around the
  same phase as eval:
  - steps `55-60`: top-5 train neighbors all open (`+1.0`);
  - step `61`: top-5 mostly open, one close;
  - step `62`: mixed, top-1 close;
  - steps `63+`: top-5 all close.
- This strongly de-prioritizes a gripper-label generation bug.
- The striking mismatch is visual: nearest training wrist/camera2 means are
  about `0.427-0.437`, while eval camera2 means are about `0.596`.

## Interpretation

This audit supports the following responsibility split:

### Less likely

- Demonstration script generated the wrong gripper sign.
- Demonstration script generated globally wrong 8D action labels.
- Dataset conversion swapped action semantics or used IK labels for joint
  policy.

### Still possible, but more subtle

- Demonstration collection may have produced a narrow teacher-forced manifold:
  successful states with wrist camera views centered/cleaner than what the
  policy later creates in closed loop.
- The policy may not recover from small early visual/action errors because
  those states are underrepresented in demonstrations.
- Scene/render settings during eval may differ enough that the same joint state
  produces a brighter wrist image.

### Most likely current cause

The closed-loop baseline fails because visual observations in eval, especially
wrist/camera2, are off the training distribution.  The learned policy then
predicts an imprecise arm grasp pose and closes while still `~11-12 cm` from
the cube.

## Practical implication

The next diagnostic should not be a full retrain yet.  A better next step is a
single-episode controlled visual-domain intervention at policy input time, for
example affine-matching eval camera2 to training camera2 mean/std.  If grasp
distance improves, that directly implicates visual-domain shift.  If it does
not improve, the remaining likely cause is closed-loop state/pose distribution
rather than simple brightness statistics.
