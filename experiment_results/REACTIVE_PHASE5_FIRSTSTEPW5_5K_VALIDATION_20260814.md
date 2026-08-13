# Reactive phase-5 first-step-weighted 5k validation (2026-08-14)

## Scope and checkpoint identity

This report evaluates the short gate run `contact_recovery_reactive_phase5_firststepw5_gate_20260813`. It is a pipeline and representation diagnostic, not a formal tactile-effect result.

Hugging Face repository:

```text
Dleshers/factory-peg-insert-conditional-recovery-v3-formal400-checkpoints
```

Evaluated checkpoints:

```text
runs/contact_recovery_reactive_phase5_firststepw5_gate_20260813/arms/
  visual/checkpoints/{002000,005000}/pretrained_model
  torque/checkpoints/{002000,005000}/pretrained_model
```

SHA-256 of `model.safetensors`:

| arm/checkpoint | SHA-256 |
| --- | --- |
| visual 2k | `4f619aa5966cff0fa3dc9147eeae5fcce3cd24f5adeea622a73590063fbbba0a` |
| visual 5k | `223aba3ef6e24e6df8b3bc5bf8c2b382ed268596d8b867b4c89b9db6bfcbd521` |
| torque 2k | `56f9423be483a0a920bd38a640e1a83d5d1eb3735f740d2aa1b8c6649afd7caf` |
| torque 5k | `1332b579499f950b2eb0b80a4a25e61672e67a00acaf22651252f45addff8bfe` |

Both arms use `action_loss_first_step_weight=5.0`, `seed=1000`, `batch_size=32`, the same corrected reactive-phase-5 dataset, and the same official SmolVLA base. The torque arm uses a newly initialized 30x7 signed-torque LSTM and does not load the legacy 1D `torque_16d_encoder.pt`.

## Reliable findings

### The loss correction improved coarse visual alignment

On the same 32 direction-balanced first phase-5 states:

| checkpoint | mean first-action XY cosine | positive states | positive sectors |
| --- | ---: | ---: | ---: |
| visual 2k | 0.176 | 21/32 | 6/8 |
| visual 5k | 0.461 | 26/32 | 8/8 |
| torque 2k | 0.089 | 19/32 | 5/8 |
| torque 5k | 0.245 | 20/32 | 7/8 |

Five independent flow samples on visual 5k have a per-seed mean cosine of `0.455 ± 0.045`. Averaging those five first actions raises the cosine to `0.629`, with 27/32 positive states and 8/8 positive sectors. This confirms that first-step weighting fixed a real part of the current-action problem, while also showing substantial flow-sampling variance.

### Torque is causally useful in the 1–2.5 mm fine-alignment band

The action audit was extended with `--xy-min-m` and `--xy-max-m`. On 32 recorded expert states in the 1–2.5 mm band:

| arm | first-action cosine | positive states | positive sectors | chunk steps 0–13 |
| --- | ---: | ---: | ---: | ---: |
| visual 5k | 0.278 | 22/32 | 5/8 | 0.138 |
| torque 5k | 0.864 | 30/32 | 8/8 | 0.565 |

A 24-state paired counterfactual keeps RGB, proprioception, checkpoint and flow seed fixed and changes only the torque window:

| torque input | mean XY cosine | positive states |
| --- | ---: | ---: |
| original chronological 30x7 | 0.804 | 23/24 |
| zero | -0.160 | 10/24 |
| causal time shuffle | 0.141 | 14/24 |

Original torque is better than zero on 23/24 states and better than causal shuffle on 18/24 states. This is strong evidence that the LSTM uses signed temporal torque structure for fine correction. It is not, by itself, a closed-loop success claim.

### The remaining bottleneck is the final clearance band

Factory uses an 8.100 mm hole and a 7.986 mm peg. The nominal radial clearance is therefore only about `0.057 mm`.

In the recorded `<1 mm` band:

| arm | first-action cosine | positive states | positive sectors |
| --- | ---: | ---: | ---: |
| visual 5k | 0.197 | 21/32 | 5/8 |
| torque 5k | 0.108 | 18/32 | 5/8 |

A diagnostic visual-coarse/torque-fine rollout reached a minimum XY error of `0.643 mm`, compared with `2.175 mm` for visual in the matched evaluation configuration, but did not insert. At `<1 mm`, the online torque policy emitted mean lateral action magnitude around `0.418`, while the expert data in that band averages about `0.044`. The resulting rim oscillation prevents descent despite a consistently downward Z command.

The 5k checkpoint therefore demonstrates a tactile representation signal and improved fine alignment, but it does not meet the strict insertion gate.

## Evaluation-chain corrections implemented

### Phase-consistent takeover

The reactive phase-5 conversion omits the fixed-timer unload labels. Evaluation must reproduce the collector's unload before policy takeover. The exact collector transition executes 15 unload actions before the first phase-5 observation; use:

```bash
--pre-takeover-unload-steps 15
```

All outputs record this value so old 0/14-step runs cannot be mixed with corrected runs.

### Variance and action-support diagnostics

`eval_factory_peg_insert_native_contact_takeover.py` now supports:

- `--inference-samples`: average independent flow samples per observation;
- `--deterministic-flow-noise`, `--flow-noise-seed`, and `--flow-noise-fixed-across-steps`;
- `--action-clip` and a separate `--fine-xy-action-clip`;
- `--coarse-policy-path` and a latched coarse-to-fine policy switch;
- `--save-traces` with signed XY pose and action traces.

These are diagnostic controls. A formal comparison must apply identical sampling and clipping to both arms and report the unclipped policy result as a sensitivity analysis.

### Same-process, same-snapshot paired evaluator

Independent Isaac launches were found to diverge before policy switching even with identical seeds and explicit flow noise: pre-switch pose differences reached approximately `0.56 mm`, already ten times the peg-hole radial clearance. Such rollouts are not valid paired causal samples.

`experiment/eval_factory_peg_insert_same_state_pair.py` replaces this with:

1. one native reset and physical contact acquisition;
2. one shared visual coarse prefix;
3. capture of robot/asset root states, joint states, velocities, controller targets, EMA actions, finite-difference caches, episode buffers and the chronological 30x7 torque window;
4. in-process restore of that snapshot for visual and torque-original branches;
5. automatic hashes for initial RGB, proprioception and the simulator snapshot;
6. rejection unless both branches start from identical audited observations and restoration error is at most `1e-6`.

Only outputs with `paired_initial_observation_identical=true` are admissible for paired success-rate statistics.

A one-pair identity smoke (`seed=20261211`, one branch action) now passes: state, both RGB frames and the 30x7 torque window have identical hashes in both branches, and the maximum restore error is `1.1921e-7`. RTX did not reproduce pixel-identical images when each restored branch rendered its own first frame, so the corrected evaluator renders the fork RGB once and replays that exact first RGB to both policies; subsequent images are live and branch-specific. The smoke's 0/1 strict result is intentionally not a performance statistic.

Local smoke artifact:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/evaluation_results/
  reactive_phase5_firststepw5_gate_validation_20260813/
  same_state_pair_smoke_seed20261211_r3.json
```

## Interpretation boundary

The evidence currently supports:

- the corrected current-action loss improves visual coarse localization;
- the torque LSTM uses real temporal torque information;
- torque is markedly stronger than visual on recorded 1–2.5 mm fine-alignment states;
- current 5k policies remain unreliable below 1 mm and have not demonstrated strict insertion.

It does not yet support a headline claim that tactile input improves closed-loop strict insertion. That claim requires the same-snapshot benchmark, a stronger `<1 mm` policy, and the formal paired evaluation specified in the training handoff.
