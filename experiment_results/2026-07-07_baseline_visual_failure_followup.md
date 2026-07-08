# 2026-07-07 baseline visual failure follow-up

This note records the additional local validation performed after the 1-episode
baseline diagnostics.  No formal training, batch evaluation, dataset overwrite,
or checkpoint overwrite was performed.

## Current conclusion

The baseline checkpoint is locally loadable and produces plausible 8D joint
actions, but in closed loop it repeatedly closes the gripper before the gripper
is physically at the cube.  The strongest current evidence points to a
vision-conditioned grasp-pose error, especially from the wrist camera visual
distribution/composition, rather than a global action-scale or gripper-command
sign bug.

## Environment/control-mode check

Relevant files inspected:

- `_runtime/remote_handoff_gripper_lstm_work/experiment/run_eval_server.sh`
- `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/scripts/eval_server.py`
- `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/config/franka/__init__.py`
- `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/config/franka/pick_place_basket_joint_tacex_env_cfg.py`
- `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/config/franka/pick_place_basket_ik_rel_tacex_env_cfg.py`

Findings:

- `run_eval_server.sh` defaults to `CONTROL_MODE=joint`.
- In that default path it launches:
  `Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0`.
- `Joint-TacEx` inherits the TacEx visual/tactile scene configuration and
  overrides only the arm action with absolute 7-DoF joint targets:
  `JointPositionActionCfg(scale=1.0, use_default_offset=False)`.
- The camera configuration is inherited from `IK-Rel-TacEx`:
  - `wrist_cam`: mounted at `{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam`,
    `224x224`, pinhole focal length `24.0`, offset
    `pos=(0.11, 0.0, -0.12)`,
    `rot=(-0.70614, 0.03701, 0.03701, -0.70614)`, `ros`.
  - `table_cam`: `{ENV_REGEX_NS}/table_cam`, `224x224`, focal length `18.0`,
    offset `pos=(1.0, 0.0, 0.6)`,
    `rot=(0.35355, -0.61237, -0.61237, 0.35355)`, `ros`.
- Data collection records `rgb_table` and `rgb_wrist` from the same sensor names.
- Eval uses rename map:
  `rgb_table -> observation.images.camera1`,
  `rgb_wrist -> observation.images.camera2`.

Interpretation: a gross IK-vs-joint action semantic mismatch is unlikely for
the most recent diagnostics, assuming `CONTROL_MODE` was not manually overridden.
The remaining camera issue is more likely visual distribution/composition or
closed-loop state distribution shift, not a different camera config class.

## Closed-loop evidence

Three local single-episode diagnostics are now consistent:

| Run | n_action_steps | Seed | Success | Min EEF-cube | First negative gripper | EEF-cube at first negative | Physical close | EEF-cube at physical close | Cube effect |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_1ep_eef_seed1000_20260707_0643` | 50 | 1000 | false | 0.07436 m | 61 | 0.07723 m | 66 | 0.07568 m | cube lifted briefly by ~0.0113 m, then dropped |
| `baseline_1ep_eef_n1_seed1000_20260707` | 1 | 1000 | false | 0.08044 m | 57 | 0.09070 m | 60 | 0.08044 m | max displacement ~0.0040 m, no stable grasp |
| `baseline_1ep_eef_seed1001_20260707` | 1 | 1001 | false | 0.11249 m | 59 | 0.11709 m | 62 | 0.11291 m | essentially no cube motion |

This rules out 50-step chunking as the sole cause.  With one-step replanning,
the policy still starts closing several centimeters too far from the cube.

## Policy input parity

The saved eval-time policy snapshots show correct key/shape plumbing:

- `observation.state`: 9D normalized joint state.
- `action`: 8D absolute joint target + gripper command.
- `observation.images.camera1`: `[1, 3, 224, 224]`, values in `[0, 1]`.
- `observation.images.camera2`: `[1, 3, 224, 224]`, values in `[0, 1]`.
- `observation.tactile.force_grid` is present but baseline does not use
  gripper torque window.

Dataset metadata remains consistent with the expected authority:

- 200 episodes
- 41,276 frames
- state `[9]`
- action `[8]`
- two RGB cameras `[3,224,224]`
- gripper torque window `[30,1]`

## Image distribution evidence

From 600 sampled training frames vs eval snapshots:

- Training `camera1` mean median: `0.73323`.
- Eval `camera1` mean: about `0.70894-0.72058`, near the low tail of sampled
  training frames.
- Training `camera2` mean median: `0.44160`; p95 `0.48815`; p99 `0.49310`.
- Eval `camera2` mean: about `0.57367-0.59616`, above nearly all sampled
  training wrist frames.

Visual inspection also suggests eval wrist images contain a much larger bright
robot-body region and show the cube near the image edge, while sampled training
wrist frames are generally more gripper/table/cube centered.

## Offline perturbation evidence

Script:

- `remote_workspace/experiment/offline_visual_perturbation_audit.py`

Outputs:

- `experiment_results/2026-07-07_visual_perturbation_audit/visual_perturbation_rows.csv`
- `experiment_results/2026-07-07_visual_perturbation_audit/visual_perturbation_summary.json`

On 40 mid/grasp-phase training samples:

- `cam2_eval_affine`: mean arm drift L2 `0.04446`, max `0.20250`.
- `cam2_eval_mean`: mean arm drift L2 `0.05016`, max `0.18069`.
- `cam2_zero`: mean arm drift L2 `0.05716`, max `0.27045`.
- `cam1_eval_affine`: mean arm drift L2 `0.04393`.
- No variant flipped the gripper command sign.

Interpretation: visual perturbations move the predicted arm target by a
meaningful amount while leaving open/close timing sign mostly intact.  This
matches the closed-loop symptom: the gripper command is semantically correct,
but the grasp pose is wrong.

## Eval snapshot replay/ablation evidence

Script:

- `remote_workspace/experiment/replay_eval_snapshots_ablation.py`

Outputs:

- `experiment_results/2026-07-07_eval_snapshot_ablation/snapshot_ablation_rows.csv`
- `experiment_results/2026-07-07_eval_snapshot_ablation/snapshot_ablation_summary.json`
- `experiment_results/2026-07-07_eval_snapshot_ablation/snapshot_ablation_compact_summary.json`

The replayed original snapshots reproduce the recorded eval actions closely:

- mean arm L2 vs recorded: `0.02607`
- max arm L2 vs recorded: `0.04954`
- mean gripper delta vs recorded: `0.00871`

Therefore the snapshot replay path is reliable enough for local ablations.

At seed1001 step 60, where the recorded gripper command has turned negative:

| Variant | Arm drift L2 vs original | Predicted gripper | Sign |
| --- | ---: | ---: | ---: |
| `orig` | 0.00000 | -1.01382 | -1 |
| `cam2_train_affine` | 0.15557 | -0.98917 | -1 |
| `cam2_zero` | 0.14670 | -0.98280 | -1 |
| `cam2_flat_train_mean` | 0.07924 | -1.01776 | -1 |
| `cam1_train_affine` | 0.02597 | -1.02040 | -1 |
| `both_train_affine` | 0.02615 | -1.02275 | -1 |
| `swap_cameras` | 0.06973 | -0.99255 | -1 |

This is the sharpest evidence so far: near grasp closure, wrist-camera changes
can move the 7D arm target by `0.15` L2 while preserving the gripper close sign.

## Less likely or partially ruled out causes

- Action dimensionality mismatch: unlikely.  Action is 8D and Joint-TacEx uses
  absolute 7D joint target plus binary gripper.
- Gripper command sign bug: unlikely.  The command turns negative and physical
  gripper qpos closes accordingly.
- Image key swap: unlikely as a sole cause.  The documented rename map matches
  table/wrist semantics; explicit camera swap changes actions but does not
  explain the observed systematic premature close by itself.
- `n_action_steps=50`: not sufficient.  The same failure appears with
  `n_action_steps=1`.
- Checkpoint load/normalizer corruption: unlikely.  Offline training-sample
  inference has low loss and perfect gripper sign accuracy on sampled rows.

## Remaining possible factors

Ranked by current evidence:

1. Wrist-camera visual distribution/composition shift in eval.
   The wrist image brightness and object framing differ strongly from sampled
   training frames, and wrist ablations produce large arm-target drift at grasp
   time.
2. Closed-loop compounding error.
   The policy may be trained on successful scripted trajectories and may not
   recover when its own earlier joint errors put the wrist/gripper in a slightly
   off-manifold pose.
3. Gripper state out-of-distribution after closure.
   In eval, gripper qpos reaches `0.0`, below the sampled training state minimum
   seen in the parity audit.  This likely affects recovery after closure, but it
   does not explain why the first close happens too far from the cube.
4. Object-pose/phase distribution.
   The seed1000/seed1001 cube initial poses fall within the configured reset
   ranges, but the local persistent LeRobot dataset does not contain raw
   cube/basket pose metadata.  Confirming whether failed eval initial states
   match successful-demo phase distributions requires raw HDF5 metadata or
   newly instrumented collection/eval rollouts.
5. Lighting/material/rendering differences.
   The high wrist brightness could be caused by camera composition, lighting,
   material, or exposure differences.  Camera config class appears shared, so
   this should be investigated as scene/render distribution rather than sensor
   naming first.

## Recommended next command

Run one more single-episode diagnostic that saves policy snapshots at denser
steps around approach and closure, using an absolute trajectory path to avoid
the previous nested relative path issue:

```bash
mkdir -p "$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707"
PORT=5566 CONTROL_MODE=joint \
TRAJECTORY_LOG="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/baseline_eef_trajectory.jsonl" \
bash _runtime/remote_handoff_gripper_lstm_work/experiment/run_eval_server.sh visual
```

Then, in a second shell:

```bash
HF_HOME="$PWD/_runtime/remote_handoff_gripper_lstm_work/.cache/huggingface" \
XDG_CACHE_HOME="$PWD/_runtime/remote_handoff_gripper_lstm_work/.cache" \
LEROBOT_EVAL_SNAPSHOT_DIR="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707/policy_input_snapshots" \
LEROBOT_EVAL_SNAPSHOT_STEPS="0,10,20,30,40,50,55,57,59,60,61,62,65,70,80,100,120" \
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/lerobot-eval \
  --policy.path=_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000 \
  --policy.load_vlm_weights=false \
  --policy.n_action_steps=1 \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost --env.server_port=5566 \
  --env.task=pick_place --env.torque_window_size=30 \
  --env.observation_height=224 --env.observation_width=224 \
  --env.control_mode=joint \
  --rename_map='{"observation.images.rgb_table":"observation.images.camera1","observation.images.rgb_wrist":"observation.images.camera2"}' \
  --env.include_gripper_torque_window=false \
  --eval.n_episodes=1 --eval.batch_size=1 \
  --seed=1002 \
  --output_dir=_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707
```

If the same early-close distance recurs, the next targeted experiment should be
a controlled visual-domain intervention: either align eval wrist-camera
appearance to training distribution or retrain/fine-tune with more diverse
wrist visual states.  Do not start that experiment without explicit approval.
