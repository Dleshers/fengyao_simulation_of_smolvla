# Visual/torque common-failure root cause and fix (2026-08-13)

## Decision

The current `contact_recovery_padfix_gate_20260813` 5k checkpoints are valid
pipeline-gate checkpoints, but they are not suitable for a formal tactile
comparison.  Do not extend their closed-loop evaluation and do not warm-start
formal training from them.

The common failure is caused by a hidden-timer label structure plus underfit 5k current-action mapping. It is not caused by camera calibration, state/action scaling,
torque-window construction, `action_is_pad`, random seeds, or the Factory
controller.  Exact native replay matched state and oracle action with zero
error, matched the 30x7 torque window after the replay correction, and matched
both cameras to sub-pixel numerical differences.

## Root cause

The old contact-recovery training view includes a fixed-duration phase-4
prefix before lateral recentering.  Phase is audit metadata and is not a policy
input.  From a nearly unchanged rim-contact observation, a one-frame policy
therefore cannot infer when the fixed timer expires.

Full 50-step action-chunk audit on eight held source demonstrations showed:

| arm | XY cosine, steps 0--13 | XY cosine, steps 14--32 |
| --- | ---: | ---: |
| visual | -0.139 | +0.301 |
| torque-original | +0.438 | +0.811 |

The corrective direction is encoded mainly in the later chunk positions.

A second audit selected the exact first phase-5 recenter frame. The old 5k visual checkpoint reached 6/8 positive sectors with mean first-action cosine 0.392; torque-original reached only 5/8 with mean 0.296. Therefore removing the hidden-timer prefix is necessary but not sufficient: the 5k gate checkpoints also underfit the current-action mapping and must be retrained from the base.
With `n_action_steps=1`, the model repeatedly replans and executes only the
unreliable first position.  Executing 20 or 40 actions open loop and temporal
averaging were tested as diagnostics; neither was stable across arms and
directions, so these are not accepted fixes.

## Implemented data fix

`convert_factory_peg_insert_hdf5_to_lerobot.py` now accepts
`--policy-phase-min`.  For the legacy balanced64 gate data, use:

```bash
--policy-label-only --policy-phase-min 5 --torque-dim 7
```

This keeps the real chronological 30x7 torque history, skips the
fixed-timer-only phase-4 prefix, and makes the first supervised action a
state-responsive lateral recenter action.

The converted local gate dataset is:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/
  Dleshers/factory-peg-insert-contact-recovery-v1-7d-reactive-phase5-local
```

Audit result:

- 64 episodes, 2,390 frames, 15 Hz;
- all finite, signed 30x7 torque retained;
- eight sectors remain balanced at eight trajectories each;
- 64/64 first XY actions point toward the hole centre;
- mean first-action centre cosine `0.9976`, minimum `0.9683`;
- `action_is_pad` regression: 4 tests passed.

## Required retraining gate

Train visual and torque-original independently from the same padding-fixed
official SmolVLA base, with identical split, seed, sampler, optimizer and
steps.  Do not reuse a 5k checkpoint.  Keep `n_action_steps=1` during
closed-loop evaluation.

Before any large closed-loop evaluation, run
`audit_native_contact_action_chunks.py` on held demonstrations and require:

- first predicted XY action has positive centre-direction cosine in at least
  6/8 sectors for visual and 7/8 sectors for torque-original;
- mean first-action cosine is at least 0.35 for visual and 0.55 for torque;
- original torque improves first-action cosine over zero/shuffle on the same
  states;
- no systematic positive-depth ejection or grasp drift in a 2+2 smoke run.

Only after these gates pass should the paired 8+8 and then 32+32 closed-loop
evaluation begin.  The formal v4 dataset still supersedes this 64-trajectory
gate set for the final tactile claim.
