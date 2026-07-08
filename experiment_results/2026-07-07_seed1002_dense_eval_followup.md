# 2026-07-07 seed1002 dense baseline eval follow-up

This run follows the previous recommendation: one single-episode baseline eval
with dense policy input snapshots around the approach/close transition.  No
training, batch evaluation, dataset overwrite, or checkpoint overwrite was
performed.

## Run configuration

- Policy:
  `_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- Eval output:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707`
- Server:
  `Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0`
- Control mode:
  `joint`
- Policy eval:
  `n_action_steps=1`
- Seed:
  `1002`
- Dense snapshot steps:
  `0,10,20,30,40,50,55,57,59,60,61,62,65,70,80,100,120`

Artifacts:

- trajectory JSONL:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/baseline_eef_trajectory.jsonl`
- compact trajectory summary:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/eef_seed1002_dense_trajectory_summary.json`
- video:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`
- dense policy snapshots:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/policy_input_snapshots/`
- offline snapshot ablation:
  `experiment_results/2026-07-07_eval_snapshot_ablation_seed1002_dense/`

## Result

The episode failed:

- success: `false`
- reward sum: `0.0`
- episode length: `300` steps
- initial EEF-cube distance: `0.24733 m`
- minimum EEF-cube distance: `0.11495 m` at step `64`
- first negative gripper command: step `63`
- EEF-cube distance at first negative command: `0.11939 m`
- physical gripper close (`qpos < 0.005`): step `66`
- EEF-cube distance at physical close: `0.11622 m`
- cube max displacement: `2.59e-7 m`
- cube max lift: `9.50e-8 m`
- cube final displacement: `3.09e-8 m`

This is the clearest failure so far: the cube was essentially untouched.

## Close-window trajectory

| Step | Gripper cmd | Gripper qpos min | EEF-cube after | Cube z | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 55 | 0.96757 | 0.039999 | 0.14110 | 0.02200 | open, approaching |
| 57 | 0.97008 | 0.039998 | 0.13319 | 0.02200 | open |
| 59 | 0.94824 | 0.040000 | 0.12713 | 0.02200 | open |
| 60 | 0.93769 | 0.040000 | 0.12466 | 0.02200 | open |
| 61 | 0.90270 | 0.040000 | 0.12202 | 0.02200 | open |
| 62 | 0.90890 | 0.040000 | 0.11939 | 0.02200 | open |
| 63 | -0.98592 | 0.029662 | 0.11725 | 0.02200 | close command begins |
| 64 | -1.01344 | 0.019465 | 0.11495 | 0.02200 | closest approach |
| 65 | -0.99400 | 0.009422 | 0.11516 | 0.02200 | closing |
| 66 | -1.00880 | 0.003456 | 0.11622 | 0.02200 | physically closed |
| 70 | -1.00510 | 0.000063 | 0.12213 | 0.02200 | closed, moving away |

The arm never reached a contact-quality pose.  Closing starts with roughly
`11.7-11.9 cm` EEF-cube separation, and the cube z coordinate stays unchanged.

## Dense snapshot image statistics

The saved policy input snapshots have the expected keys and image tensor shapes.
The camera2/wrist brightness remains in the same shifted range observed in
seed1001:

- number of snapshots: `17`
- camera2 mean range: `0.57386-0.59621`
- previous sampled training camera2 mean median: `0.44160`
- previous sampled training camera2 mean p99: `0.49310`

Thus seed1002 independently reproduces the high-brightness wrist observation
distribution seen in earlier failed evals.

## Snapshot ablation

Offline replay was run with:

```bash
remote_workspace/experiment/replay_eval_snapshots_ablation.py
```

Output:

- `experiment_results/2026-07-07_eval_snapshot_ablation_seed1002_dense/snapshot_ablation_rows.csv`
- `experiment_results/2026-07-07_eval_snapshot_ablation_seed1002_dense/snapshot_ablation_summary.json`
- `experiment_results/2026-07-07_eval_snapshot_ablation_seed1002_dense/snapshot_ablation_compact_summary.json`

Important caveat:

Around the gripper transition boundary, replayed `orig` actions do not perfectly
match the recorded closed-loop actions.  In particular, replay can flip gripper
sign around steps `60-62`.  Therefore, the trajectory JSONL remains the
authority for gripper timing.  The ablation is still useful for measuring local
visual sensitivity of the arm action.

Selected arm drift results near the close window:

| Step | Variant | Arm drift L2 vs orig | Gripper sign |
| ---: | --- | ---: | ---: |
| 55 | `cam2_train_affine` | 0.03723 | open |
| 55 | `cam2_zero` | 0.03110 | open |
| 57 | `cam2_zero` | 0.05802 | open |
| 59 | `cam2_zero` | 0.08461 | open |
| 60 | `cam2_train_affine` | 0.06148 | close in replay |
| 60 | `cam2_zero` | 0.05505 | open |
| 60 | `swap_cameras` | 0.10473 | open |
| 70 | `cam2_train_affine` | 0.04834 | close |
| 70 | `swap_cameras` | 0.05587 | close |

Across all 17 snapshots:

- `cam2_train_affine`: mean arm drift L2 `0.02815`, max `0.06148`
- `cam2_zero`: mean arm drift L2 `0.03652`, max `0.08461`
- `cam2_flat_train_mean`: mean arm drift L2 `0.03596`, max `0.08922`
- `cam1_train_affine`: lower mean arm drift than camera2 variants
- `swap_cameras`: can reach arm drift above `0.10` around step 60

This is consistent with the previous seed1001 conclusion: the policy is
visually sensitive around grasp, and wrist/table visual perturbations move the
arm target while the closed-loop failure itself remains an imprecise grasp pose.

## Updated interpretation

Seed1002 makes the failure mode more specific:

1. The policy approaches the cube but stops/turns around before contact-quality
   alignment.
2. It issues close only after the EEF is still about `11-12 cm` from the cube.
3. The gripper physically closes, so command plumbing and gripper sign are not
   the main bug.
4. The cube does not move measurably, so this is not a weak grasp or slip; it is
   a missed grasp.
5. The wrist camera remains strongly shifted relative to sampled training
   frames, and offline ablations confirm visual sensitivity of the 7D arm
   target.

Current strongest hypothesis remains:

> Baseline failure is driven by closed-loop visual policy grasp-pose error,
> with wrist-camera visual distribution/composition shift as the leading
> contributor.  Action dimension, joint-control mode, gripper sign, and
> checkpoint/normalizer corruption are now de-prioritized.

## Recommended next action

Do not start training yet.  The next most informative controlled test is one of:

1. a single-episode eval with a deliberate wrist-camera visual/domain alignment
   intervention; or
2. a replay/diagnostic comparing successful training-phase wrist frames against
   eval frames at similar EEF-cube distances and joint states.

The first option changes the evaluation observation distribution and should be
done only as a diagnostic patch, not as a hidden benchmark result.
