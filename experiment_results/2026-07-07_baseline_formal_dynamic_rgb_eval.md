# Baseline formal evaluation after dynamic-RGB and terminal fixes

Date: 2026-07-07

## Fixes active

- Isaac eval no longer forces `use_fabric=False`; table and wrist RGB follow
  live scene geometry.
- terminal success is read from IsaacLab's latched `termination_manager` term,
  before post-step reset geometry can corrupt the result.
- telemetry records all termination terms, pre-step cube/basket positions, and
  whether post-step scene values came from auto-reset.

## Formal configuration

- model: `smolvla-franka-pickplace-baseline-50k-seed1000`;
- 10 episodes, seed 1000, randomized task reset;
- joint control, 8D absolute joint/gripper action;
- state 9D, two live RGB cameras at 224x224;
- `n_action_steps=1`; no fixed cube and no gripper gate.

Output:

`_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_formal_dynamic_rgb_n10_seed1000_20260707/`

## Result

- success: `0/10` (`0%`);
- all episodes completed 300 steps;
- runtime: `907.07 s`, mean `90.71 s/episode`;
- no `success`, `cube_dropping`, or `basket_dropping` termination fired;
- all ten MP4 files contain 300 frames and live robot motion.

Per-episode minimum EEF-cube distance (m):

`[0.0669, 0.0548, 0.0587, 0.0434, 0.0710, 0.0511, 0.0604, 0.0383, 0.0670, 0.0624]`

Per-episode maximum cube height (m):

`[0.0223, 0.0252, 0.0220, 0.0336, 0.0220, 0.0335, 0.0229, 0.0379, 0.0221, 0.0220]`

Episode 7 produced the most object motion (`0.1514 m`) but only raised the cube
to `0.0379 m`; this is push/drag or unstable contact, not a stable lift.  Most
episodes either miss the cube or make weak contact.  Thus the corrected formal
randomized evaluation remains 0%, but it is now a valid visual closed-loop
measurement.  The earlier fixed-cube diagnostic remains important: at a central
pose the same policy can grasp, lift and transport, so randomized object-pose
coverage/final approach generalization is the leading failure factor.

## Known operational warning

Isaac Sim 4.5 still aborts during shutdown in `TiledCamera.__del__` with a weak
reference error.  This occurs after videos, JSONL and metrics are flushed and
does not invalidate the completed evaluation.
