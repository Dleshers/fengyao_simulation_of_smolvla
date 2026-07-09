# 2026-07-09 updated experiment goal: gated torque injection

## Current conclusion

The pick-and-place task is visually solvable enough that it is not a strong
benchmark for demonstrating tactile benefit.  It should be used as an
integration and non-collapse sanity check:

- the torque/LSTM path can train;
- checkpoints can be saved and loaded;
- evaluation can run closed-loop;
- actions do not numerically explode;
- the task does not collapse to 0% success.

The current ungated torque-LSTM continuation satisfies non-collapse but hurts
performance:

| run | success |
|---|---:|
| visual50k | 8/10 |
| visual50k + visual-only 5k | 8/10 |
| visual50k + ungated torque-LSTM 5k | 7/10 |

The matched visual-only 5k control keeps the exact same success/failure pattern
as visual50k, while ungated torque adds one failure.  Therefore the next
engineering goal is not to prove tactile benefit on pick-and-place, but to
make the torque injection non-disruptive.

## New immediate goal

Run a gated/zero-init torque-LSTM 5k continuation:

- base checkpoint: targeted5 visual 50k;
- dataset: targeted5 clean;
- frozen torque encoder;
- torque adapter zero-initialized;
- scalar torque gate initialized to `1.0`;
- same eval seeds as previous runs.

Important correction: a first gate-0 smoke run showed that setting both the
adapter and scalar gate to exactly zero creates a dead torque branch.  The final
checkpoint kept `model.torque_gate`, `model.torque_to_expert.weight`, and
`model.torque_to_expert.bias` all at zero.  Therefore the meaningful gated
ablation uses zero-init adapter with a nonzero scalar gate.  This still starts
with zero torque perturbation, because the adapter output is zero, but the
adapter receives gradients immediately.

Expected interpretation:

- `>= 8/10`: gated/zero-init prevents the observed torque perturbation;
- `7/10`: the issue is not only initial adapter magnitude;
- `< 7/10`: gated implementation or torque conditioning is harming the policy;
- `> 8/10`: torque may provide benefit even on this visually solvable task.

## Formal downstream goal

Use pick-and-place only as a feasibility / non-regression benchmark.  The main
tactile-benefit comparison should move to a contact-rich task where vision alone
cannot reliably infer contact, slip, grasp stability, or force state.

Official or larger LeRobot datasets should be treated as a later pretraining
ablation, not a prerequisite for this non-collapse check.

## Added scripts and patches

- `experiment/train_visual50k_plus_gated_torque_5k.sh`
- `experiment/eval_gated_torque_n10.sh`
- `patches/2026-07-09_lerobot_gated_zero_init_torque.patch`

The patch adds:

- `policy.torque_zero_init_adapter`
- `policy.torque_gate_init`
- `model.torque_gate`

Defaults preserve old behavior unless explicitly enabled.
