# 2026-07-08 fixed-cube baseline diagnostics and data-expansion assessment

## Scope

Continued local validation of the pure SmolVLA baseline after the camera refresh
fix.  No training, checkpoint overwrite, dataset overwrite, upload, or broad
batch evaluation was started.

Primary question: if the remaining baseline failure is caused by data, should
the training set be expanded?

## New diagnostic runs

### 1. Fixed-cube 3x3 grid, unmodified baseline policy

Output directory:

`_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_fixedcube_grid_3x3_seed1200_20260708/`

Artifacts:

- `trajectory.jsonl`
- `eval_info.json`
- `videos/isaaclab_tactile_remote_0/eval_episode_{0..8}.mp4`
- `physical_episode_summary.json`
- `physical_episode_summary.csv`

Cube poses covered the configured reset rectangle:

- x: `0.35, 0.425, 0.50`
- y: `-0.18, -0.115, -0.05`
- z: `0.022`

LeRobot aggregate metric reported `0/9`.

Physical re-analysis from server telemetry:

| Result | Count |
| --- | ---: |
| terminal physical success | 0/9 |
| cube lifted above 4.5 cm | 2/9 |
| cube entered basket XY bounds | 2/9 |
| no meaningful lift | 7/9 |

Important details:

- The two partial successes were the low-x poses:
  - `(0.35, -0.18, 0.022)`
  - `(0.35, -0.115, 0.022)`
- In both, the cube was grasped/lifted and brought near or into basket XY, but
  terminal release/open did not satisfy success.
- The other seven poses generally approached and closed, but did not lift the
  cube.
- Minimum EEF-cube distance was sometimes small (`~3.9-6.9 cm`), so the robot
  is not frozen and the action schema is not globally broken.  The failure is
  around robust grasp alignment and post-grasp release.

### 2. Fixed-cube 3 episodes with ground-truth close gate

Output directory:

`_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_fixedcube_gate055_n3_seed1300_20260708/`

Artifacts:

- `trajectory.jsonl`
- `eval_info.json`
- `videos/isaaclab_tactile_remote_0/eval_episode_{0..2}.mp4`
- `physical_episode_summary.json`
- `physical_episode_summary.csv`

This diagnostic forced the gripper command open unless
`EEF-cube <= 0.055 m`.  It is not a baseline score; it tests whether early close
timing alone explains failure.

LeRobot aggregate metric reported `0/3`.

Physical re-analysis:

| Pose | Physical outcome |
| --- | --- |
| `(0.425, -0.18, 0.022)` | contact/drag, no lift |
| `(0.425, -0.115, 0.022)` | lifted and reached basket XY, but no release/open success |
| `(0.50, -0.05, 0.022)` | no lift |

Interpretation:

- Close timing is part of the problem, but not the only problem.
- Gate helped one central pose reach a much better grasp/place trajectory, but
  it did not fix release or broad pose robustness.
- This points to missing closed-loop coverage near contact, lift, transport, and
  release, not a simple gripper sign bug.

## Server telemetry bug found and fixed in runtime workspace

While parsing the first 3x3 run, a telemetry bug became clear:

- on manual max-step timeout, `eval_server.py` set `done=True`;
- it then trusted `termination_manager.success` even when the environment had
  not actually emitted a terminal success;
- `post_step_scene_was_auto_reset` was also set from `done`, so timeout frames
  could be mislabeled as auto-reset frames.

Runtime file patched:

`_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/scripts/eval_server.py`

Change:

- distinguish `env_done` from `hit_max_steps`;
- only trust `termination_terms["success"]` when `env_done` is true;
- record `env_step_done` and `hit_max_steps`;
- set `post_step_scene_was_auto_reset` from `env_done`, not from manual timeout.

Validation:

- `python3 -m py_compile .../eval_server.py` passed.
- `git diff --check` on touched runtime scripts passed.

This does not change the already-recorded videos, but future JSONL success
fields should be less misleading.

## Current answer: is this a data problem?

The current evidence supports a data/distribution problem in the broad sense,
but not a corrupted-data or demo-label bug.

Less likely now:

- wrong 8D action schema;
- gripper sign inversion;
- stale camera frames;
- model/checkpoint not loading;
- robot not moving;
- pure LeRobot metrics as the sole explanation.

More likely:

1. The 200 successful demonstrations do not provide enough robust closed-loop
   coverage across cube pose, wrist-camera geometry, and recovery states.
2. The policy reproduces plausible temporal phases, but closes or releases from
   spatially imperfect states.
3. The low-x side of the reset range is partially learned; central/high-x and
   high-y poses are much weaker.
4. The release/open phase is under-controlled: even when the cube reaches basket
   XY, terminal release success is not achieved.

## Should the dataset be expanded?

Yes, if the goal is to make the baseline or torque-LSTM branch succeed under the
current randomized reset distribution.  But it should be targeted expansion,
not just more of the same.

Recommended data additions:

1. Stratified cube-pose coverage over the full reset rectangle:
   - x: at least low/center/high;
   - y: at least low/center/high;
   - include yaw variation.
2. Extra demonstrations for central/high-x poses where the current baseline
   approaches but fails to lift.
3. Recovery/correction demonstrations:
   - slightly missed grasp;
   - cube pushed a few cm;
   - gripper closed near but not centered;
   - re-approach before lift.
4. Post-grasp and release-focused demonstrations:
   - cube already lifted;
   - cube near basket;
   - open/release timing;
   - retreat after release.
5. Save object and basket pose metadata alongside future collections, even if
   it is not fed to the policy.  This will make distribution audits much more
   direct.

Do not immediately retrain on an expanded dataset until the next collection
script records pose metadata and the eval-server timeout/success telemetry fix
is included in the reproducible patch set.

## Suggested next command

Archive the runtime eval-server changes into a small patch for review:

```bash
git diff --no-index /dev/null _runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/scripts/eval_server.py
```

Then rerun a tiny fixed-grid smoke eval after applying the telemetry patch to
confirm timeout episodes now report `is_success=False` in JSONL.
