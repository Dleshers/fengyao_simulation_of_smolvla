# 2026-07-07 fixed-cube-pose baseline diagnostic

This note records a controlled single-episode baseline eval with the cube reset
to a fixed, training-range center pose.

No training, batch evaluation, dataset overwrite, or checkpoint overwrite was
performed.

## Purpose

Previous audits showed that failed eval observations can be robot-state-near to
training data while still having different cube-relative wrist geometry.  This
diagnostic tests whether moving the cube from the failing seed1002 pose to a
more central training-range pose improves the grasp approach.

## Runtime diagnostic patch

I added a default-off Isaac eval-server diagnostic argument:

```text
--fixed-cube-pose x,y,z[,qw,qx,qy,qz]
```

and connected it through:

```text
FIXED_CUBE_POSE=...
```

in `run_eval_server.sh`.

The first run exposed a wiring mistake: the argument was parsed but not passed
from `main()` into `IsaacLabEnvWrapper`, so the cube was not actually changed.
I fixed the wiring and reran into a separate directory.

Patch archive:

- `patches/2026-07-07_eval_server_fixed_cube_diagnostic.patch`

## Effective run

Output directory:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_center_effective_seed1002_20260707
```

Video:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_center_effective_seed1002_20260707/videos/isaaclab_tactile_remote_0/eval_episode_0.mp4
```

Trajectory:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_center_effective_seed1002_20260707/baseline_eef_trajectory.jsonl
```

Policy snapshots:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_center_effective_seed1002_20260707/policy_input_snapshots
```

Configuration:

| Field | Value |
| --- | --- |
| Policy | baseline SmolVLA 50k seed1000 |
| Eval seed | `1002` |
| Control mode | joint |
| n_action_steps | `1` |
| Fixed cube pose | `[0.425, -0.115, 0.022, 1, 0, 0, 0]` |
| Episodes | `1` |

The trajectory includes:

```json
{"event":"diagnostic_fixed_cube_pose_applied","fixed_cube_pose_env":[0.4250000119,-0.1150000021,0.0219999999,1.0,0.0,0.0,0.0]}
```

so the override was active in the effective run.

## Result

The episode still failed:

| Metric | Original seed1002 | Fixed cube center |
| --- | ---: | ---: |
| Success | `false` | `false` |
| Reset cube position | `[0.4994, -0.1137, 0.0220]` | `[0.4250, -0.1150, 0.0220]` |
| Min EEF-cube distance | `0.11495 m` | `0.06059 m` |
| Step of min distance | `64` | `97` |
| First negative gripper cmd | step `63` | step `63` |
| EEF-cube at first negative cmd | `0.11725 m` | `0.06975 m` |
| First physical close | step `64` | step `65` |
| EEF-cube at physical close | `0.11495 m` | `0.06551 m` |
| Max cube displacement | `~0.0000003 m` | `0.04988 m` |
| Max cube lift | `~0.00000009 m` | `0.00015 m` |

Key step samples from the fixed-cube run:

| Step | EEF-cube | Gripper cmd | Mean gripper qpos | Cube position |
| ---: | ---: | ---: | ---: | --- |
| 55 | `0.09868` | `+0.96793` | `0.039999` | `[0.4250, -0.1150, 0.0220]` |
| 60 | `0.07574` | `+0.93417` | `0.040000` | `[0.4250, -0.1150, 0.0220]` |
| 63 | `0.06975` | `-0.99042` | `0.030082` | `[0.4275, -0.1160, 0.0219]` |
| 65 | `0.06551` | `-0.99004` | `0.014551` | `[0.4291, -0.1164, 0.0219]` |
| 80 | `0.06135` | `-1.04184` | `0.003698` | `[0.4502, -0.1208, 0.0218]` |
| 100 | `0.06079` | `-1.02752` | `0.001181` | `[0.4609, -0.1248, 0.0219]` |
| 300 | `0.22759` | `-1.04581` | `0.000001` | `[0.4726, -0.1298, 0.0220]` |

## Interpretation

The fixed-cube intervention changed the failure mode:

- Original seed1002 barely touched the cube.
- Fixed-center cube led to clear contact/pushing: cube displacement increased
  to about `5 cm`.
- However, the cube was not lifted and success remained `0%`.

This supports the object-pose / relative-geometry mismatch hypothesis: the
baseline is sensitive to cube placement, and putting the cube in a more central
pose helps the arm reach contact distance.

But fixed pose is not sufficient.  The learned approach still closes from a
pose that pushes/slides the cube instead of enclosing and lifting it.  The
remaining failure is therefore more specifically a grasp-quality / recovery
problem:

- gripper close phase is plausible;
- gripper physically closes;
- cube contact occurs in the fixed-center case;
- contact is not centered or vertically aligned enough to lift;
- the policy keeps closing and pushing rather than correcting.

## Updated diagnosis

Current evidence still does not point to a hard demonstration-label bug.  The
stronger diagnosis is:

1. object pose and wrist-camera relative geometry strongly affect behavior;
2. the baseline has weak closed-loop recovery once its approach is offset;
3. the dataset may contain successful teacher-forced grasps but not enough
   near-contact correction / failed-contact recovery states;
4. the learned policy has insufficient grasp-quality precision for stable
   cube capture.

## Suggested next command

The next most useful single-episode diagnostic is to keep the fixed-center cube
but delay or gate gripper closure until the EEF is closer to the cube.  This
separates "arm cannot reach a grasp pose" from "gripper closes too early".

Do not run it as a formal evaluation.  Use it only as a controlled intervention
with trajectory logging.
