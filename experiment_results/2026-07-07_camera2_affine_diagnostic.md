# 2026-07-07 camera2 affine visual diagnostic

This is a single-episode diagnostic intervention, not a benchmark result.  It
tests whether matching eval wrist/camera2 mean and std to training camera2
statistics is sufficient to improve the failed baseline grasp.

No training, batch evaluation, dataset overwrite, or checkpoint overwrite was
performed.

## Patch

Runtime LeRobot eval was patched with a default-off environment variable:

- runtime file:
  `_runtime/remote_handoff_gripper_lstm_work/lerobot-tactile/src/lerobot/scripts/lerobot_eval.py`
- archive:
  `patches/2026-07-07_lerobot_eval_camera2_affine_diagnostic.patch`

Switch:

```bash
LEROBOT_EVAL_CAMERA2_TRAIN_AFFINE=1
```

It applies only after policy preprocessing and before `policy.select_action`.
It modifies `observation.images.camera2` in the policy input only:

```text
camera2 = (camera2 - current_mean) / current_std * train_camera2_std + train_camera2_mean
```

Default target:

- mean: `0.441604882478714`
- std: `0.206113621592522`

## Run

- baseline policy:
  `_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- output:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1002_cam2affine_20260707`
- seed: `1002`
- control mode: `joint`
- `n_action_steps=1`
- episodes: `1`

## Intervention check

The intervention did what it was supposed to do numerically.  Saved policy
snapshots show:

| Step | camera2 mean | camera2 std |
| ---: | ---: | ---: |
| 0 | 0.44160 | 0.20611 |
| 55 | 0.44164 | 0.20604 |
| 60 | 0.44164 | 0.20604 |
| 62 | 0.44164 | 0.20604 |
| 63 | 0.44164 | 0.20604 |
| 64 | 0.44164 | 0.20604 |
| 65 | 0.44164 | 0.20604 |
| 66 | 0.44164 | 0.20604 |
| 70 | 0.44164 | 0.20604 |

So the previous high mean `~0.596` was normalized down to the sampled training
camera2 statistics.

## Result comparison with original seed1002

| Metric | Original seed1002 | camera2 affine seed1002 |
| --- | ---: | ---: |
| success | false | false |
| min EEF-cube distance | 0.11495 m | 0.11431 m |
| min-distance step | 64 | 64 |
| first negative gripper cmd | step 63 | step 63 |
| EEF-cube at first negative cmd | 0.11939 m | 0.11909 m |
| physical close step | 66 | 66 |
| EEF-cube at physical close | 0.11622 m | 0.11661 m |
| cube max displacement | 2.59e-7 m | 2.59e-7 m |
| cube max lift | 9.50e-8 m | 9.50e-8 m |

Close-window trajectory for affine run:

| Step | Gripper cmd | Gripper qpos min | EEF-cube after | Cube z |
| ---: | ---: | ---: | ---: | ---: |
| 55 | 0.96651 | 0.039998 | 0.14496 | 0.02200 |
| 57 | 0.96429 | 0.039998 | 0.13547 | 0.02200 |
| 59 | 0.96555 | 0.040000 | 0.12808 | 0.02200 |
| 60 | 0.94517 | 0.040000 | 0.12513 | 0.02200 |
| 61 | 0.93962 | 0.040000 | 0.12202 | 0.02200 |
| 62 | 0.94271 | 0.040000 | 0.11909 | 0.02200 |
| 63 | -0.99259 | 0.029654 | 0.11673 | 0.02200 |
| 64 | -1.01935 | 0.019454 | 0.11431 | 0.02200 |
| 65 | -0.98830 | 0.009410 | 0.11518 | 0.02200 |
| 66 | -1.00675 | 0.003452 | 0.11661 | 0.02200 |
| 70 | -1.00652 | 0.000063 | 0.12465 | 0.02200 |

## Interpretation

Matching camera2 global mean/std alone is not enough.  The policy still misses
the cube by about `11-12 cm`, and the cube remains essentially untouched.

This refines the visual-shift hypothesis:

- camera2 brightness/statistics are a symptom and a useful detector of eval
  distribution shift;
- but the sufficient failure factor is probably not simple global brightness;
- remaining likely factors are:
  - wrist-camera composition/framing/geometric content;
  - cube visibility/location in wrist view;
  - closed-loop state drift away from successful demonstration manifold;
  - object pose distribution near the edge of the demonstrated success region;
  - possible scene/render/material differences that alter more than global
    intensity.

## Updated answer to the demonstration-script question

This diagnostic further weakens the hypothesis that demonstration labels are
simply wrong.  In nearest-neighbor dataset states, gripper labels switch from
open to close at a similar phase, and the camera2 affine intervention does not
rescue the trajectory.  The more likely issue is that the baseline policy has
not learned a robust closed-loop visual grasp policy for the eval observation
manifold.

The demonstration generation pipeline can still be implicated at the data
coverage level: it may have produced successful, teacher-forced trajectories
without enough off-manifold recovery or enough wrist-view diversity.  That is a
data distribution/coverage problem, not a direct action-label bug.
