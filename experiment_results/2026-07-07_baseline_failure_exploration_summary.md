# 2026-07-07 baseline closed-loop failure exploration summary

## Scope

This document summarizes the local baseline validation and failure-cause exploration performed after restoring the SmolVLA + gripper torque-LSTM workspace.

Constraints observed:

- No full training was started.
- No batch evaluation was started.
- Existing datasets/checkpoints were not overwritten.
- No Hugging Face token or private credential was printed.
- Runtime data, videos, and logs were kept under ignored `_runtime/`.

## Repository and machine state

- Git commit: `e5391140f34d1e58de26e37d2be5467f87dfe37b`
- Host: `ubuntu2204-System-Product-Name`
- GPU check outside sandbox: NVIDIA GeForce RTX 3060, driver `570.211.01`, 12288 MiB total, about 320 MiB used after eval cleanup
- Disk: `/dev/sda3`, 916G total, 799G available
- Baseline checkpoint:
  `_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- Dataset:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1`

## Environment and patches used

- LeRobot env:
  `_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot`
- Isaac env:
  `_runtime/remote_handoff_gripper_lstm_work/.conda/isaaclab`
- `pyzmq==27.1.0` is installed in both envs.
- Runtime eval server supports `TRAJECTORY_LOG`.
- Runtime eval server was patched to log eef telemetry:
  `patches/2026-07-07_eval_server_eef_telemetry.patch`
- LeRobot safetensors checkpoint-save fallback patch archived:
  `patches/2026-07-07_lerobot_safetensors_save_fix.patch`
- `SmolVLAConfig.tactile_token_mode` compatibility patch is present in:
  `remote_handoff_gripper_lstm/lerobot_overrides/configuration_smolvla.py`

## Offline baseline validity

The uploaded pure SmolVLA baseline is locally loadable and numerically consistent with the dataset.

Offline audit artifacts:

- `experiment_results/2026-07-07_baseline_local_inference_audit/summary.json`
- `experiment_results/2026-07-07_baseline_local_inference_audit/samples.csv`
- `experiment_results/2026-07-07_baseline_local_inference_audit/README.md`

Key results over 600 sampled dataset frames:

- mean forward loss: `0.022345`
- arm L2 error mean: `0.05612`
- gripper sign accuracy: `1.0000`
- gripper binary accuracy: `1.0000`
- gripper MAE: `0.00904`

Interpretation:

- The checkpoint is not corrupted.
- The action normalizer/postprocessor is active.
- The local model predicts the dataset gripper command sign correctly.
- The action dimension and local policy output are correct: 8D action.

## Static action/observation audit

Confirmed from local bridge/config inspection:

- eval observation state is `[arm_joint_pos(7), gripper_qpos(2)]`, matching training `observation.state: [9]`;
- visual keys are mapped as required:
  `rgb_table -> observation.images.camera1`,
  `rgb_wrist -> observation.images.camera2`;
- joint-mode action is forwarded unchanged after policy postprocessing;
- dataset action is 8D:
  `[joint_pos_target_abs(7), gripper_cmd(1)]`;
- gripper sign semantics are consistent with Isaac joint binary gripper:
  positive opens, negative closes.

Strong configuration risk found:

- baseline config uses `chunk_size=50`, `n_action_steps=50`;
- at 20 Hz this means about 2.5 seconds of queued open-loop actions before the policy re-observes.

This is risky, but later testing shows it is not the only cause.

## Closed-loop diagnostic episodes

### 1. Baseline 1-episode probe, default action chunking

Output:

- `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_seed1000_20260707_0629/`

Result:

- success: `false`
- reward sum: `0.0`
- episode length: `300` steps
- first negative gripper command: step `61`
- physical gripper close: step `66`
- cube final displacement: `0.00344 m`
- cube max displacement: `0.02340 m`
- cube max lift: `0.01098 m`
- final cube-basket XY distance: `0.26882 m`

Interpretation:

- The gripper does close.
- The cube is nudged/slightly lifted but not stably grasped or transported.

### 2. Baseline 1-episode probe with eef telemetry, default action chunking

Output:

- `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1000_20260707_0643/`

Result:

- success: `false`
- reward sum: `0.0`
- episode length: `300` steps
- initial eef-cube distance: `0.25557 m`
- min eef-cube distance: `0.07436 m` at step `63`
- first negative gripper command: step `61`
- eef-cube distance at first negative command: `0.07723 m`
- physical gripper close: step `66`
- eef-cube distance at physical close: `0.07568 m`
- cube final displacement: `0.00396 m`
- cube max displacement: `0.02369 m`
- cube max lift: `0.01132 m`

Interpretation:

- The policy closes while still about `7.5 cm` from the cube center.
- This points to approach/grasp-pose/timing error rather than gripper sign or scale inversion.

### 3. Baseline 1-episode probe with `n_action_steps=1`

Output:

- `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_n1_seed1000_20260707/`

Artifacts:

- `baseline_eef_n1_trajectory.jsonl`
- `eef_n1_trajectory_summary.json`
- `videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`
- sampled frames: `frame_001.jpg` ... `frame_005.jpg`

Result:

- success: `false`
- reward sum: `0.0`
- episode length: `300` steps
- initial eef-cube distance: `0.25557 m`
- min eef-cube distance: `0.08044 m` at step `60`
- first negative gripper command: step `57`
- eef-cube distance at first negative command: `0.09070 m`
- physical gripper close: step `60`
- eef-cube distance at physical close: `0.08044 m`
- cube final displacement: `0.00020 m`
- cube max displacement: `0.00397 m`
- cube max lift: `0.00258 m`
- joint target tracking error mean: `0.03711`

Interpretation:

- Fully reactive `n_action_steps=1` did not rescue the policy.
- It approached no closer than about `8 cm` and barely moved the cube.
- Therefore the long 50-step queue is not the sole root cause. It may exacerbate failures, but the first-step/closed-loop visual policy itself is already selecting a premature or spatially offset grasp.

## Consolidated failure diagnosis

Current best conclusion:

The pure SmolVLA baseline is valid offline but fails closed-loop because its approach/grasp pose is wrong under the local Isaac evaluation distribution. The gripper closes, but it closes while the end-effector remains roughly `7-9 cm` from the cube center. The cube is at most nudged or slightly lifted and is never transported toward the basket.

## 2026-07-07 follow-up: eval input parity audit

A fourth single-episode diagnostic was run at seed `1001` with `n_action_steps=1` and policy-input snapshots enabled:

- output: `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1001_20260707/`
- report: `experiment_results/2026-07-07_eval_input_parity_audit.md`
- distribution artifacts: `experiment_results/2026-07-07_eval_input_parity_audit/`
- snapshot patch: `patches/2026-07-07_lerobot_eval_input_snapshot.patch`

Result:

- success: `false`
- min eef-cube distance: `0.11249 m`
- first negative gripper command: step `59`
- eef-cube distance at first negative command: `0.11709 m`
- physical close: step `62`
- eef-cube distance at physical close: `0.11291 m`
- cube displacement: effectively zero

New evidence:

- policy receives the expected keys and shapes:
  - `observation.images.camera1`: `[1,3,224,224]`
  - `observation.images.camera2`: `[1,3,224,224]`
  - `observation.state`
  - 8D action
- image values are correctly scaled to `[0,1]`;
- arm joint state remains inside training min/max and within about `1.54σ`;
- closed gripper qpos reaches `0.0`, below the dataset state minimum around `0.017`, so post-close state is OOD;
- more importantly, the visual distribution is shifted:
  - eval table-camera mean is around the lowest `0.3%` of the sampled training distribution;
  - eval wrist-camera mean is around the highest `99.5%` of the sampled training distribution;
  - visual inspection shows eval wrist frames dominated by a large white robot-body region with cube near the edge, unlike typical training wrist frames.

Updated diagnosis:

- camera key/shape/value-range is correct;
- robot arm state is not grossly OOD before grasp;
- the strongest remaining root cause is wrist-camera render/extrinsic/composition distribution shift, possibly combined with general closed-loop imitation fragility.

Ruled out or strongly de-prioritized:

- corrupted baseline checkpoint;
- missing action normalizer/postprocessor;
- wrong local action dimensionality;
- gripper command sign inversion;
- missing `pyzmq`;
- long `n_action_steps=50` as the sole cause.

Still plausible:

- camera/render distribution mismatch between dataset collection and eval;
- subtle camera key/preprocessing mismatch not visible from shape/key audit;
- object initial pose/task reset mismatch;
- visual policy closed-loop compounding error near grasp;
- action target semantics are correct globally but insufficiently precise for contact-rich grasping.

## Runtime issue observed

Each local Isaac eval server run completed the episode and wrote trajectory/video data, but aborted during cleanup with:

```text
ReferenceError: weakly-referenced object no longer exists
```

The traceback is in Isaac camera/replicator/tiled-camera cleanup. This is currently treated as a cleanup-path issue because the eval client ended normally and artifacts were saved.

## Recommended next step

Run one more single-episode, non-batch diagnostic at a different eval seed to determine whether the premature `7-9 cm` close distance is systematic:

```bash
PORT=5565 CONTROL_MODE=joint \
TRAJECTORY_LOG=_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1001_20260707/baseline_eef_trajectory.jsonl \
bash _runtime/remote_handoff_gripper_lstm_work/experiment/run_eval_server.sh visual
```

Then, in a second shell:

```bash
HF_HOME=_runtime/remote_handoff_gripper_lstm_work/.cache/huggingface \
XDG_CACHE_HOME=_runtime/remote_handoff_gripper_lstm_work/.cache \
CONTROL_MODE=joint \
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/lerobot-eval \
  --policy.path=_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000 \
  --policy.load_vlm_weights=false \
  --policy.n_action_steps=1 \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost --env.server_port=5565 \
  --env.task=pick_place --env.torque_window_size=30 \
  --env.observation_height=224 --env.observation_width=224 \
  --env.control_mode=joint \
  --rename_map='{"observation.images.rgb_table":"observation.images.camera1","observation.images.rgb_wrist":"observation.images.camera2"}' \
  --env.include_gripper_torque_window=false \
  --eval.n_episodes=1 --eval.batch_size=1 \
  --seed=1001 \
  --output_dir=_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1001_20260707
```

If seed `1001` repeats the same premature-close distance, the next useful investigation is image/preprocessing parity: save synchronized eval observations at policy input time and compare them against dataset frames from similar state/action phases.

## Follow-up after seed1001 snapshot/ablation audit

Additional local-only analysis is recorded in:

- `experiment_results/2026-07-07_baseline_visual_failure_followup.md`
- `experiment_results/2026-07-07_eval_input_parity_audit.md`
- `experiment_results/2026-07-07_visual_perturbation_audit/`
- `experiment_results/2026-07-07_eval_snapshot_ablation/`

Key update:

- `seed1001`, `n_action_steps=1`, also failed; the physical gripper close
  occurred at an EEF-cube distance of about `0.1129 m`, with essentially no
  cube motion.
- The eval server script defaults to `CONTROL_MODE=joint`, so the latest local
  diagnostics used `Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0` unless that
  environment variable was manually overridden.
- `Joint-TacEx` inherits the same table/wrist camera configuration from the
  TacEx scene cfg and only overrides the arm action to absolute 7D joint
  targets. This de-prioritizes an IK-vs-joint action mismatch.
- Eval policy input snapshots have correct keys, image shapes, value ranges,
  9D state, and 8D action.
- Eval wrist-camera brightness/composition remains strongly shifted from
  sampled training frames: sampled train `camera2` mean median is about
  `0.4416`, while eval snapshots are about `0.5737-0.5962`.
- Offline perturbation and eval-snapshot ablation both show that wrist-camera
  changes can move arm joint targets substantially while preserving gripper
  sign. At seed1001 step 60, `cam2_train_affine` changes the 7D arm target by
  L2 `0.1556` while the gripper prediction remains close/negative.

Updated strongest hypothesis:

The baseline does not primarily fail because of checkpoint corruption, action
scale, gripper sign, or action chunking. It most likely fails because the visual
policy predicts an imprecise grasp pose under closed-loop eval observations,
with the wrist-camera visual distribution/composition shift as the leading
contributor.

## Follow-up after seed1002 dense snapshot eval

Additional single-episode dense diagnostic:

- `experiment_results/2026-07-07_seed1002_dense_eval_followup.md`
- output directory:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_dense_20260707`
- seed: `1002`
- `n_action_steps=1`
- dense policy snapshots at steps:
  `0,10,20,30,40,50,55,57,59,60,61,62,65,70,80,100,120`

Key result:

- success: `false`
- minimum EEF-cube distance: `0.11495 m` at step `64`
- first negative gripper command: step `63`
- EEF-cube distance at first negative command: `0.11939 m`
- physical gripper close: step `66`
- EEF-cube distance at physical close: `0.11622 m`
- cube max displacement: `2.59e-7 m`
- cube max lift: `9.50e-8 m`

This is a clean missed-grasp episode: the policy closes while still about
`11-12 cm` from the cube, and the cube is essentially untouched.

The eval wrist-camera distribution again matches the previous failed-eval
pattern:

- seed1002 dense snapshots `camera2` mean range: `0.57386-0.59621`
- sampled training `camera2` mean median: `0.44160`
- sampled training `camera2` mean p99: `0.49310`

Offline seed1002 snapshot ablation was saved under:

- `experiment_results/2026-07-07_eval_snapshot_ablation_seed1002_dense/`

Important caveat: replayed `orig` gripper actions around steps `60-62` do not
perfectly match the recorded closed-loop gripper actions, so the trajectory JSONL
remains the authority for gripper timing.  The ablation still shows local visual
sensitivity of arm targets near the close window; for example `cam2_zero`
reaches arm-target L2 drift `0.08461` at step `59`, and camera swap reaches
`0.10473` at step `60`.

Updated conclusion after seed1002:

The baseline failure is now best described as a repeatable missed-grasp pose
under visual closed-loop control.  The gripper subsystem works, but the arm
does not reach a contact-quality pose before the close command.  Wrist-camera
visual distribution/composition shift remains the leading concrete factor to
test next.

## Follow-up: demonstration-script vs distribution-mismatch audit

Additional documents:

- `experiment_results/2026-07-07_dataset_nearest_eval_state_audit.md`
- `experiment_results/2026-07-07_camera2_affine_diagnostic.md`
- `experiment_results/2026-07-07_dataset_nearest_eval_state_audit_seed1002/`

Nearest-neighbor audit:

- selected seed1002 eval states from steps `55-70`;
- compared raw eval state `[joint_pos_after(7), gripper_qpos_after(2)]`
  against all `41,276` LeRobot dataset states;
- used z-normalized state distance from dataset stats;
- extracted top-5 nearest train actions and embedded camera statistics.

Result:

- For eval steps `55-60`, top-5 nearest training frames all have open gripper
  action (`+1.0`), matching eval.
- Step `61` is mixed but mostly open.
- Step `62` is mixed with top-1 close.
- Steps `63+` have top-5 close actions (`-1.0`), matching eval phase.

This de-prioritizes a direct demonstration-generation bug such as wrong gripper
sign, wrong action dimensionality, or wrong IK/joint labels.

However, nearest training camera2 means remain around `0.427-0.437`, while eval
camera2 means at the same key steps remain around `0.596`.  Thus similar joint
states appear with very different wrist visual inputs.

Camera2 affine diagnostic:

- added a default-off LeRobot eval switch:
  `LEROBOT_EVAL_CAMERA2_TRAIN_AFFINE=1`;
- policy-input camera2 was affine-matched to training mean/std
  (`0.44160/0.20611`);
- ran one diagnostic episode at seed `1002`.

Result:

- intervention succeeded numerically: policy input camera2 mean/std became
  about `0.44164/0.20604`;
- episode still failed;
- min EEF-cube distance changed only from `0.11495 m` to `0.11431 m`;
- first close and physical close steps remained `63` and `66`;
- cube displacement/lift remained essentially zero.

Updated interpretation:

The problem is probably not a simple global wrist brightness issue either.  The
strongest remaining explanation is a closed-loop visual grasp-pose failure under
eval observation geometry/composition/state distribution.  The demonstration
pipeline may be responsible in the weaker sense of insufficient coverage
successful teacher-forced trajectories, but current evidence does not support a
hard bug in demonstration action labels.

## Follow-up: automated train/eval distribution-gap audit

New detailed note:

- `experiment_results/2026-07-07_train_eval_distribution_gap_audit.md`

New outputs:

- `experiment_results/2026-07-07_paired_train_eval_visual_diff_seed1002/paired_visual_diff_rows.csv`
- `experiment_results/2026-07-07_paired_train_eval_visual_diff_seed1002/paired_visual_diff_summary.json`
- `experiment_results/2026-07-07_paired_train_eval_visual_diff_seed1002/paired_eval_train_montage.jpg`

The paired audit compares failed eval policy-input snapshots against rank-1
nearest training frames selected by z-normalized 9D robot/gripper state.

Results:

- 8 paired frames with snapshots.
- camera1/table difference is moderate: mean image delta `-0.0193`, L1 diff
  `0.0531`, gray correlation `0.8840`.
- camera2/wrist difference is large: mean image delta `+0.1642`, bright fraction
  delta `+0.2545`, white-ish fraction delta `+0.2624`, L1 diff `0.1851`, gray
  correlation only `0.3533`.
- The montage shows that robot-state-nearest train frames can have very
  different object-relative wrist geometry because the dataset state does not
  include cube pose.

This further supports a distribution/closed-loop mismatch explanation rather
than a direct demonstration-label corruption explanation.

## Follow-up: fixed-cube-pose baseline diagnostic

New detailed note:

- `experiment_results/2026-07-07_fixed_cube_pose_baseline_diagnostic.md`

New patch archive:

- `patches/2026-07-07_eval_server_fixed_cube_diagnostic.patch`

The eval server was given a default-off diagnostic option to reset the cube to a
fixed env-relative pose.  The effective controlled run used:

- seed `1002`;
- `n_action_steps=1`;
- fixed cube pose `[0.425, -0.115, 0.022, 1, 0, 0, 0]`;
- output directory
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_center_effective_seed1002_20260707`.

Result:

- success remained `false`;
- min EEF-cube distance improved from `0.11495 m` to `0.06059 m`;
- first negative gripper command remained step `63`;
- cube displacement increased from essentially zero to `0.04988 m`;
- cube lift remained negligible (`0.00015 m`).

Interpretation: moving the cube to a central training-range pose substantially
improves contact and changes the failure from "no meaningful contact" to
"push/slide without lift", but it does not rescue the baseline.  Object pose and
relative wrist geometry are therefore real contributors, while the remaining
issue is grasp-quality precision and lack of closed-loop recovery near contact.

## Follow-up: gripper close gate diagnostic

New detailed note:

- `experiment_results/2026-07-07_gripper_close_gate_diagnostic.md`

New patch archive:

- `patches/2026-07-07_eval_server_gripper_close_gate_diagnostic.patch`

The eval server was given a default-off diagnostic gate that suppresses negative
gripper commands until EEF-cube distance is below a chosen threshold.

Results on seed `1002` with fixed cube pose `[0.425, -0.115, 0.022]`:

- no gate: first close step `63`, min EEF-cube distance `0.06059 m`, cube slid
  about `49.9 mm`, lift negligible;
- gate `0.050 m`: raw close started at step `61`, but effective close never
  occurred because the arm never reached `5 cm`; min distance was `0.06521 m`;
- gate `0.065 m`: effective close was delayed to step `72`, min distance
  improved to `0.05521 m`, cube max lift rose to `4.66 mm`, but success remained
  false and the cube was pushed sideways.

Interpretation: premature close is a contributor, but not the sole root cause.
The arm approach itself plateaus offset from the cube, and even delayed close
does not produce stable grasp geometry.  This further narrows the baseline
failure to final approach/contact alignment and weak closed-loop recovery rather
than a simple action-label sign/scale bug.

## Critical correction: eval camera frames are stale

Inspection of
`baseline_1ep_fixedcube_gate0065_seed1002_20260707/videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`
found that the apparent lack of arm motion is real in the recorded pixels, but
not in the simulator state telemetry:

- the episode contains 300 frames at 20 FPS, but sampled table-camera frames at
  steps 0, 30, 60, 90, 120, 180, 240, and 299 retain the same robot and object
  geometry;
- policy-input snapshots from both `camera1` and `camera2` likewise retain the
  initial geometry; their numerical differences are dominated by render noise;
- meanwhile the JSONL reports an EEF net displacement of `0.16060 m`, maximum
  joint net displacement of `3.49329 rad`, and cube displacement of about
  `0.032 m`.

Therefore the physics/action telemetry is live, but the RGB sensor output sent
to the policy is stale.  These runs are not valid visual closed-loop evaluations:
the policy effectively receives the initial visual scene repeatedly while the
9D joint/gripper state continues to update.  Earlier physical observations
(approach, contact, cube push, and transient lift) remain valid descriptions of
the executed trajectory, but their attribution to normal visual closed-loop
policy behavior must be withdrawn.

Priority is now to diagnose/fix Isaac camera refresh (render scheduling, sensor
update, and scene-transform synchronization), verify motion with a deterministic
pixel/geometry freshness test, and only then repeat the no-gate and gate runs.
Further gripper-gate threshold sweeps before that fix are not informative.

## Camera refresh fix and corrected baseline rerun

Detailed note:

- `experiment_results/2026-07-07_isaac_camera_refresh_fix.md`

The stale-camera root cause was the eval server's forced `use_fabric=False`, a
configuration difference absent from collection and replay.  Restoring the task
default passed a deterministic 60-step RGB freshness test.  A corrected
one-episode baseline rerun then grasped the cube, lifted it to `0.21186 m`,
transported it `0.29422 m`, and reached the basket rim.  This is a major change
from the stale-vision push/no-lift result and establishes that frozen RGB
directly impaired policy behavior.  Earlier visual evaluation scores are
invalid and must be rerun.  The corrected run still reports failure, and the
next issue is terminal success capture occurring after IsaacLab auto-reset.

## Formal baseline after both fixes

Detailed report:

- `experiment_results/2026-07-07_baseline_formal_dynamic_rgb_eval.md`

Terminal success is now sourced from IsaacLab's latched termination-manager
term.  A new randomized 10-episode evaluation with live RGB, seed 1000 and
`n_action_steps=1` completed normally at `0/10` success.  All episodes ran 300
steps; none triggered success or object-dropping termination.  Minimum
EEF-cube distance ranged from `0.0383` to `0.0710 m`; maximum cube height was
only `0.0379 m`.  This corrected result confirms that the baseline still fails
under randomized resets, primarily at final approach/grasp acquisition, while
the central fixed-cube run shows the policy retains meaningful grasp/transport
capability in a favorable part of the observation distribution.
