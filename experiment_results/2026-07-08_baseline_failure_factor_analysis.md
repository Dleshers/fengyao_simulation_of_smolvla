# 2026-07-08 baseline failure factor analysis

## Scope

This note extends the corrected dynamic-RGB baseline evaluation analysis.

No training, checkpoint overwrite, dataset overwrite, upload, or batch re-evaluation
was started.  The analysis is read-only over the existing 10-episode formal
baseline rollout and the restored LeRobot dataset.

Repository commit:

`e5391140f34d1e58de26e37d2be5467f87dfe37b`

Primary rollout analyzed:

`_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_formal_dynamic_rgb_n10_seed1000_20260707/`

New machine-readable summaries:

- `analysis_episode_summary.json` inside the rollout directory;
- `experiment_results/2026-07-08_train_eval_failure_factor_analysis/dataset_action_timing_summary.json`;
- `experiment_results/2026-07-08_train_eval_failure_factor_analysis/eval_vs_train_first_close_state.json`.

## Formal baseline behavior after camera refresh fix

The corrected formal randomized evaluation remains:

- success: `0/10`;
- all episodes reached the 300-step limit;
- no `success`, `cube_dropping`, or `basket_dropping` termination fired.

Per-episode contact/grasp evidence:

| ep | cube0 x | cube0 y | min EEF-cube dist | horizontal at min | vertical EEF-cube at min | physical close step | max cube z | max cube XY displacement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.386 | -0.092 | 0.0669 | 0.0373 | 0.0555 | 53 | 0.0223 | 0.0004 |
| 1 | 0.456 | -0.106 | 0.0548 | 0.0114 | 0.0536 | 55 | 0.0252 | 0.0082 |
| 2 | 0.498 | -0.113 | 0.0587 | 0.0134 | 0.0572 | 57 | 0.0220 | 0.0027 |
| 3 | 0.428 | -0.115 | 0.0434 | 0.0315 | 0.0299 | 108 | 0.0336 | 0.0238 |
| 4 | 0.465 | -0.069 | 0.0710 | 0.0432 | 0.0563 | 62 | 0.0220 | 0.0004 |
| 5 | 0.425 | -0.104 | 0.0511 | 0.0078 | 0.0505 | 80 | 0.0335 | 0.0536 |
| 6 | 0.350 | -0.174 | 0.0604 | 0.0231 | 0.0558 | 69 | 0.0229 | 0.0085 |
| 7 | 0.432 | -0.123 | 0.0383 | 0.0234 | 0.0304 | 130 | 0.0379 | 0.1514 |
| 8 | 0.483 | -0.104 | 0.0670 | 0.0358 | 0.0567 | 70 | 0.0221 | 0.0004 |
| 9 | 0.465 | -0.147 | 0.0624 | 0.0247 | 0.0573 | 60 | 0.0220 | 0.0010 |

Interpretation:

- The robot moves and the gripper closes.
- The common miss mode is not a gripper sign inversion.  The commanded gripper
  switches from open to close once per episode and the physical gripper reaches
  near-zero aperture.
- Most episodes close with the EEF still about `5.5-7.1 cm` from the cube
  center.  For a 5 cm cube, that is often outside a stable grasp envelope.
- Episodes 3, 5, and 7 get closer or make stronger contact, but they push/drag
  rather than lift.  Episode 7 moved the cube `15.1 cm`, but only raised it to
  `3.79 cm`.

## Dataset action/timing comparison

The restored dataset has:

- 200 episodes;
- 41,276 frames;
- action `[8] = absolute Franka joint target [7] + gripper command [1]`;
- `fps=20`.

Dataset first negative gripper command:

- all 200 episodes contain a close transition;
- first-close frame p05/p50/p95: `68 / 71 / 73`;
- first-close episode fraction p05/p50/p95: `0.329 / 0.343 / 0.356`.

Dataset action dimension ranges are broad enough to cover the formal eval
commands.  Eval actions are not obviously saturated or dimension-swapped.  The
formal eval gripper command range is approximately `[-1.09, 1.05]`, matching
the expected binary open/close convention.

## First physical-close state comparison

The strongest new finding is that the robot joint state at first physical close
is usually not grossly out-of-distribution relative to the dataset's first-close
joint-state distribution.

Eval first physical close vs training first-close state distribution:

| ep | close step | EEF-cube dist at close | arm z-score L2 | max abs arm z |
|---:|---:|---:|---:|---:|
| 0 | 53 | 0.0674 | 1.70 | 0.96 |
| 1 | 55 | 0.0562 | 1.43 | 0.75 |
| 2 | 57 | 0.0588 | 2.85 | 1.69 |
| 3 | 108 | 0.0446 | 12.96 | 8.95 |
| 4 | 62 | 0.0710 | 2.04 | 1.02 |
| 5 | 80 | 0.0554 | 11.20 | 8.03 |
| 6 | 69 | 0.0615 | 3.38 | 2.06 |
| 7 | 130 | 0.0537 | 20.31 | 11.25 |
| 8 | 70 | 0.0679 | 2.93 | 1.69 |
| 9 | 60 | 0.0637 | 2.26 | 1.53 |

Interpretation:

- In 7/10 episodes, the policy reaches a familiar first-close arm posture.
- But the cube is not under the gripper at that familiar posture.
- This points away from action decoding, gripper semantics, or checkpoint
  corruption, and toward visual/object localization or training-distribution
  coverage.
- Episodes 3/5/7 become highly OOD later, after contact/pushing.  That is likely
  a downstream compounding-error effect, not the initial cause.

## Reset distribution and expert gates

The current task config randomizes the cube in:

- x: `0.35-0.50`;
- y: `-0.18--0.05`;
- z: `0.0203`;
- yaw: `-0.5-0.5`.

The 10 formal eval cube poses are inside that configured range.

The state-machine expert, however, uses ground-truth `cube_pos` for approach and
only proceeds through stable grasp when:

- `ee_cube_dist < 0.055`;
- cube/EEF motion is coupled;
- gripper is physically closed;
- the condition is stable for 8 frames.

The baseline's physical close distance is frequently just outside this expert
gate.  A 1-3 cm visual localization or closed-loop correction error is therefore
enough to convert a valid-looking approach into a miss.

## Current ranked hypotheses

1. Visual/object localization distribution shift or insufficient visual
   generalization.
   Evidence: previous input parity audit found table-camera mean near the low
   tail of training and wrist-camera mean near the high tail; eval wrist images
   were visually dominated by bright robot body/cube edge composition.  The
   corrected dynamic RGB now moves, but the policy still closes at familiar
   joint states while the cube is offset from the gripper.

2. Dataset coverage is too narrow for randomized closed-loop deployment.
   Evidence: training first-close posture is narrow and the policy appears to
   reproduce it; random cube poses inside the configured range still fail.
   Fixed/central-pose diagnostics remain useful because they show less severe
   behavior near favorable poses.

3. Behavioral cloning compounding error near contact.
   Evidence: after weak contact, episodes 3/5/7 drift far from the first-close
   training state distribution and push/drag the cube instead of lifting.

4. Camera timing/render parity is not fully proven, although the stale-camera
   bug is fixed.
   Evidence: eval and collection both use the same named sensors, but collection
   records camera frames before `env.step()` and labels the same-step controller
   target.  This is likely semantically valid, but a dedicated replay-vs-eval
   image parity check at matched states would be stronger.

Lower-priority causes now:

- gripper sign inversion;
- missing/incorrect action normalizer;
- wrong 8D action schema;
- stale RGB frames;
- `n_action_steps=50` as the sole cause;
- CUDA/pyzmq/server transport issues.

## Recommended next diagnostic

Run a small controlled-pose evaluation grid, not full randomized batch training:

1. fix cube poses at a 3x3 grid over the configured reset range;
2. run 1 episode per pose with `n_action_steps=1`;
3. log the same JSONL/video artifacts;
4. compute success/contact/lift versus cube pose.

This directly tests whether the current baseline only works in a small
sub-region of the reset distribution.  If the grid confirms a sub-region, the
next fix should target data/visual coverage or camera parity before torque-LSTM
training conclusions are drawn.

