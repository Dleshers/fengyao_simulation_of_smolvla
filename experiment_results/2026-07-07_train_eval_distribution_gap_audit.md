# 2026-07-07 train/eval distribution-gap audit

This note summarizes an automated offline audit comparing the training dataset
against the failed closed-loop baseline episode `seed1002`.

No training, batch evaluation, checkpoint overwrite, or dataset mutation was
performed.

## Question

Can we analyze the relationship between the training set and the actual
closed-loop situation?

Yes.  I automated two complementary checks:

1. nearest-neighbor lookup in training data by robot observation state;
2. paired visual-difference analysis between saved eval policy-input snapshots
   and the nearest training frames.

The goal is to test whether the baseline fails because the demonstration labels
are directly wrong, or because closed-loop eval observations are outside the
training distribution.

## Inputs

- Dataset:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1`
- Baseline checkpoint:
  `_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- Failed dense diagnostic eval:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707`
- Eval snapshots:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/policy_input_snapshots`

## Scripts

- `remote_workspace/experiment/dataset_nearest_eval_state_audit.py`
- `remote_workspace/experiment/paired_train_eval_visual_diff_audit.py`

## Outputs

Nearest-state audit:

- `experiment_results/2026-07-07_dataset_nearest_eval_state_audit_seed1002/nearest_rows.csv`
- `experiment_results/2026-07-07_dataset_nearest_eval_state_audit_seed1002/nearest_summary.json`
- `experiment_results/2026-07-07_dataset_nearest_eval_state_audit_seed1002/nearest_camera_montage.jpg`

Paired visual-difference audit:

- `experiment_results/2026-07-07_paired_train_eval_visual_diff_seed1002/paired_visual_diff_rows.csv`
- `experiment_results/2026-07-07_paired_train_eval_visual_diff_seed1002/paired_visual_diff_summary.json`
- `experiment_results/2026-07-07_paired_train_eval_visual_diff_seed1002/paired_eval_train_montage.jpg`

## Method

For key failed eval steps around the close-command transition, I matched each
eval observation state against all dataset states using z-normalized L2
distance.  The state vector is the dataset's 9D robot/gripper state:

- 7 arm joint positions;
- 2 gripper joint positions.

It does not include cube pose.

For rank-1 nearest training frames with available eval snapshots, I then
compared:

- camera1/table RGB statistics;
- camera2/wrist RGB statistics;
- brightness, white-ish fraction, saturation, edge density, and coarse spatial
  layout;
- eval/train image L1/L2/correlation;
- side-by-side visual montage.

## Key quantitative result

Eight eval/train pairs had both nearest-state matches and saved eval snapshots.

Aggregate state/action context:

| Metric | Result |
| --- | --- |
| Paired frames | 8 |
| z-normalized state distance | mean `0.876`, min `0.340`, max `3.107` |
| eval EEF-cube distance | mean `0.1256 m`, min `0.1152 m`, max `0.1411 m` |
| eval gripper command range | `[-1.005, 0.970]` |
| nearest train gripper action range | `[-1.0, 1.0]` |

Camera1/table is comparatively close:

| Metric | Mean delta / score |
| --- | --- |
| camera1 mean delta | `-0.0193` |
| camera1 white-ish fraction delta | `-0.0505` |
| camera1 L1 diff | `0.0531` |
| camera1 gray correlation | `0.8840` |

Camera2/wrist is strongly shifted:

| Metric | Mean delta / score |
| --- | --- |
| camera2 mean delta | `+0.1642` |
| camera2 std delta | `+0.0720` |
| camera2 bright fraction delta | `+0.2545` |
| camera2 white-ish fraction delta | `+0.2624` |
| camera2 saturated-region centroid x delta | `+0.0576` |
| camera2 saturated-region centroid y delta | `-0.0823` |
| camera2 L1 diff | `0.1851` |
| camera2 gray correlation | `0.3533` |

The wrist camera gap is therefore much larger than the table camera gap.

## Visual/geometric interpretation

The paired montage shows the most important qualitative difference:

- Eval camera2 often sees the cube far toward the upper part of the wrist image,
  while the gripper is still spatially offset from it.
- The nearest training frame by robot state often has the target object much
  closer to the gripper or in a very different wrist-image location.
- Because the 9D state contains robot/gripper state but not cube pose, a close
  robot-state match does not guarantee a close end-effector-to-object or
  wrist-image geometry match.

This is consistent with the measured closed-loop failure: the gripper closes at
approximately the expected phase, but while the EEF is still about `7-12 cm`
from the cube in previous diagnostics, and `~11.5-14.1 cm` in the paired frames
used here.

## Answer: demo-generation script or something else?

The current evidence does not support a hard demonstration-generation bug such
as:

- wrong action dimensionality;
- wrong gripper sign;
- corrupted gripper open/close labels;
- baseline checkpoint/normalizer incompatibility;
- camera key swap.

The nearest-state audit shows that the training data contains similar robot
states with a similar open-to-close phase.  Around eval steps `63+`, nearest
training labels are mostly close actions, matching the eval policy's close
transition.  That weakens the hypothesis that the demonstration script simply
wrote the wrong gripper command.

The stronger explanation is a distribution/closed-loop mismatch:

- closed-loop eval reaches robot states whose relative cube geometry differs
  from successful demonstrations;
- wrist/camera2 visual composition is substantially off-distribution;
- the policy closes at a plausible temporal phase but from a spatially offset
  grasp pose;
- simple global camera2 mean/std correction did not fix the behavior, so the
  gap is not just brightness.

The demonstration pipeline may still be implicated in a weaker data-generation
sense: the collected successful demonstrations may not cover enough object
poses, visual appearances, and recovery/off-manifold states for closed-loop
control.  But this is different from a direct script bug in the saved labels.

## Current most likely failure factors

Ranked by current evidence:

1. Wrist/camera2 geometric/content distribution gap.
2. Closed-loop state drift: small early pose errors move the policy away from
   the successful demonstration manifold.
3. Object pose or relative EEF-cube pose mismatch not represented in the 9D
   state nearest-neighbor metric.
4. Dataset coverage issue: successful demos may be too narrow or too
   teacher-forced to support recovery.
5. Remaining lower-probability possibilities: subtle camera extrinsic/render
   mismatch, visual preprocessing detail, or action scaling interaction.

## Recommended next command

Rerun the paired audit if snapshots or nearest rows are regenerated:

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
  remote_workspace/experiment/paired_train_eval_visual_diff_audit.py \
  --nearest-csv experiment_results/2026-07-07_dataset_nearest_eval_state_audit_seed1002/nearest_rows.csv \
  --snapshot-dir _runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/policy_input_snapshots \
  --output-dir experiment_results/2026-07-07_paired_train_eval_visual_diff_seed1002_rerun \
  --rank 1
```

If the next intervention is allowed, the most informative controlled eval would
be to reset the cube/object pose to a known training-like pose and run one
single-episode baseline eval.  Success there would strongly indicate eval
initial-state/object-pose distribution mismatch; failure would shift attention
back toward action scaling or deeper policy/preprocessing issues.
