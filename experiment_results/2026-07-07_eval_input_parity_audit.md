# 2026-07-07 eval input parity audit

## Scope

This audit continued the baseline failure investigation by saving the actual policy inputs used during a local closed-loop eval episode and comparing them against the training dataset distribution.

No training or batch evaluation was started. One additional single-episode diagnostic was run.

## Diagnostic run

- Policy: `_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- Eval seed: `1001`
- `n_action_steps`: `1`
- Server port: `5565`
- Output dir:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1001_20260707`
- Policy input snapshots:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1001_20260707/policy_input_snapshots`
- Server trajectory actual path:
  `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1001_20260707/baseline_eef_trajectory.jsonl`

The trajectory path was nested because `TRAJECTORY_LOG` was passed as a relative path and the server runs from `IsaacLab-Tactile`.

## Result

- success: `false`
- reward sum: `0.0`
- episode length: `300`
- initial eef-cube distance: `0.22610 m`
- minimum eef-cube distance: `0.11249 m` at step `61`
- first negative gripper command: step `59`
- eef-cube distance at first negative command: `0.11709 m`
- physical gripper close: step `62`
- eef-cube distance at physical close: `0.11291 m`
- cube final displacement: `0.00000003 m`
- cube max displacement: `0.00000006 m`

This seed reproduced the same failure mode more strongly than seed `1000`: the policy closed the gripper while still about `11 cm` from the cube center, and the cube essentially did not move.

## Policy input snapshot checks

Snapshots were saved at steps:

- `0`
- `20`
- `40`
- `57`
- `60`
- `80`
- `120`

Each snapshot contains:

- policy observation keys;
- normalized `observation.state` summary;
- `observation.images.camera1` PNG and stats;
- `observation.images.camera2` PNG and stats;
- postprocessed 8D action.

Confirmed:

- policy receives `observation.images.camera1`;
- policy receives `observation.images.camera2`;
- both images are `[1,3,224,224]`;
- image values are float in `[0,1]`;
- policy action is 8D;
- gripper command becomes negative around the expected close phase.

Therefore the obvious key/shape/value-range failures are not present.

## Image distribution comparison

Training dataset sample:

- 600 frames sampled deterministically across `41,276` frames.
- Output:
  `experiment_results/2026-07-07_eval_input_parity_audit/image_distribution_summary.json`
- CSV:
  `experiment_results/2026-07-07_eval_input_parity_audit/dataset_image_stats_samples.csv`
- Montage:
  `experiment_results/2026-07-07_eval_input_parity_audit/dataset_camera_montage.jpg`

Camera 1, table camera:

- training sampled mean:
  - p05 `0.72619`
  - p50 `0.73323`
  - p95 `0.74452`
- eval snapshots:
  - about `0.70894` to `0.72058`
  - around the lowest `0.3%` of the sampled training distribution

Camera 2, wrist camera:

- training sampled mean:
  - p05 `0.42150`
  - p50 `0.44160`
  - p95 `0.48815`
  - p99 `0.49310`
- eval snapshots:
  - about `0.57367` to `0.59616`
  - around the highest `99.5%` of the sampled training distribution

Interpretation:

- The eval table camera is somewhat darker/lower-mean than typical training frames.
- The eval wrist camera is much brighter than the training distribution.
- Visual inspection suggests the eval wrist view contains a large white robot-body region and places the cube toward the image edge, while typical training wrist frames are more gripper/table/cube centered.

This is now a high-priority suspected contributor: wrist camera render/extrinsic/composition mismatch, not merely a tensor shape or key issue.

## State distribution comparison

Using dataset `meta/stats.json` and seed-1001 trajectory raw joint states:

- arm joint dimensions remain within training min/max throughout;
- max absolute z-score for arm joints is about `1.54`;
- after gripper closes, gripper qpos reaches `0.0`, while training `observation.state` gripper qpos minimum is about `0.017`.

Interpretation:

- robot arm state distribution is not the main cause of the approach failure;
- closed-gripper state becomes out-of-training-range after physical closure, which may worsen late recovery;
- this does not explain why the policy closes while still `~11 cm` from the cube, because that error appears before/at the close transition.

## Updated diagnosis

The strongest currently supported failure cause is now:

1. pure SmolVLA checkpoint is valid offline;
2. state/action schemas are correct;
3. gripper sign and command scale are correct;
4. but closed-loop eval visual observations, especially the wrist camera, are shifted from training distribution;
5. the visual policy predicts a premature/spatially offset grasp and closes far from the cube.

The `n_action_steps=50` queue is still a risk, but `n_action_steps=1` failures on seeds `1000` and `1001` show that action chunking is not the sole root cause.

## Recommended next validation

Use absolute `TRAJECTORY_LOG` paths for future runs.

Next best diagnostic:

1. Locate the exact camera config used during HDF5 collection/conversion.
2. Compare it against the current eval server `rgb_wrist` camera prim/extrinsics/FOV.
3. Run one 1-episode eval with the wrist camera corrected or disabled/blanked as an ablation.

If a quick ablation is desired before changing Isaac camera config, run a policy-input perturbation test offline:

- load saved eval snapshot images;
- lower wrist camera brightness/contrast toward training p50;
- feed the same state/table image with original vs adjusted wrist image through the policy;
- compare first-step action and gripper close timing.

This would test whether the policy is sensitive to the wrist camera distribution shift without starting another Isaac rollout.
