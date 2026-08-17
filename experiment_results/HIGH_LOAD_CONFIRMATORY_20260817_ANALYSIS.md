# Independent high-load confirmation and training-step decision (2026-08-17)

## Evaluation result

This is a new, independent, high-load-only evaluation. It contains 64 valid same-snapshot pairs, with 8 offset directions and 8 pairs per direction. All 64 pairs passed state, RGB, and 30x7 torque-history identity audits.

- Visual strict insertion: 45/64 (70.3%).
- Original torque strict insertion: 55/64 (85.9%).
- Difference: **+15.6 percentage points**.
- Paired unique wins: torque 15, visual 5, ties 44.
- Exact paired McNemar p-value: 0.0414.
- Torque was non-inferior in 7/8 directions.

The result independently reproduces the earlier high-load positive trend and supports the conditional statement: *under high-load near-hole physical contact, causal original torque information improves strict insertion success relative to vision-only control.*

## Safety and latency qualification

- Visual: 0 ejections, 2 pass-through events.
- Torque: 1 ejection, 1 pass-through event.
- Torque mean successful trajectory length: about 56.5 steps; visual: about 45.9 steps.

The torque branch therefore has higher success but slower recovery and a different safety-failure profile. The strict predeclared safety-noninferiority gate is not fully passed because ejections increased from 0 to 1. This does not erase the success-rate result, but it prevents claiming an unconditional improvement.

## Whether to train beyond 10k

Both evaluated policies are exactly 10,000-step checkpoints. The terminal 10k training logs show low and still decreasing learning rates (about `2.5e-6`) and low terminal losses (visual about `0.014`, torque about `0.005`), with no NaN or divergence evidence.

For the primary question—whether tactile torque information can improve high-load insertion—additional training is **not required before reporting the current result**. The independent 64-pair test already provides the causal comparison, and changing the checkpoint now would confound that conclusion.

Additional training is justified only as a separately identified optimization experiment targeting the remaining weaknesses: torque ejection and recovery latency. If run, preserve the current 10k models, continue both arms from the same 10k checkpoints with identical data order and seed policy, and evaluate 15k or 20k on a fresh high-load set. Do not replace the 10k result or pool checkpoints post hoc.

## Reproducibility artifacts

- Raw independent result: `experiment_results/HIGH_LOAD_CONFIRMATORY_20260817_RESULTS.json`.
- Runtime result and log: `persistent/evaluation_results/high_load_confirmatory_20260817/hard80_10k_high_load64_r1/`.
- Experiment protocol: `experiment_results/HIGH_LOAD_TACTILE_EVALUATION_PROTOCOL_20260817.md`.
