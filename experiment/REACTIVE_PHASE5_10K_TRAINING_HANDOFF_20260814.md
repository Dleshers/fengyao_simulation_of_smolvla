# Reactive phase-5 next-training handoff (2026-08-14)

## Mandatory reading and decision

Read this document after:

- `experiment/CONTACT_RECOVERY_A100_TRAINING_HANDOFF.md`;
- `experiment/CONTACT_RECOVERY_V4_DATASET_DESIGN.md`;
- `experiment_results/REACTIVE_PHASE5_FIRSTSTEPW5_5K_VALIDATION_20260814.md`.

This document is the latest instruction for the next A100 run. The downloaded 5k checkpoints are diagnostic gates. Do not publish them as the formal tactile comparison and do not start an unmodified 20k–50k run.

## Why another training revision is needed

The 5k visual policy learned the 4–7 mm coarse direction, while torque-original learned a strong causal correction in the 1–2.5 mm band. Both remain unstable in the physical-clearance region below 1 mm. More importantly, current-policy rollouts enter off-demonstration combinations of RGB, contact load and pose and then emit lateral actions roughly ten times the expert magnitude.

Adding more copies of the old direct-recovery trajectories is not an adequate fix. The next dataset must contain oracle recovery from states actually visited by the 5k policies, especially overshoot and rim-oscillation states.

## Code and environment preflight

Use the Git commit containing this handoff. Install the SmolVLA overrides exactly as described in the main A100 handoff and require all regression tests to pass.

Required invariants:

- `action_is_pad` is the only padding key used by the loss;
- `action_loss_first_step_weight=5.0` is applied to both arms;
- contact conversion uses `--policy-label-only --policy-phase-min 5 --torque-dim 7`;
- evaluation always uses `n_action_steps=1`;
- phase-5 evaluation uses exactly `--pre-takeover-unload-steps 15`;
- the legacy 1D `trained_lstm_weights/torque_16d_encoder.pt` is never loaded by the 7D arm.

Before training, run:

```bash
PYTHONPATH="$LEROBOT_ROOT/src" "$PY" -m pytest -q \
  remote_handoff_gripper_lstm/lerobot_overrides/test_smolvla_torque_lstm.py
python -m py_compile \
  experiment/audit_native_contact_action_chunks.py \
  experiment/audit_torque_counterfactual_actions.py \
  experiment/eval_factory_peg_insert_native_contact_takeover.py \
  experiment/eval_factory_peg_insert_same_state_pair.py
```

## Targeted data supplement

Keep the existing audited trajectories, but add a rollout-aggregation supplement generated from the 5k visual and torque policies. Oracle pose is used only to generate labels and audit metadata; it must never enter policy observations.

For each saved sample, restore the policy-visited state, retain the live RGB and preceding chronological 30x7 torque history, and let the recovery oracle produce a bounded unload/recenter/insert action. Save only attempts that reach and hold strict insertion.

The supplement must balance the following cells by trajectory rather than frame:

| XY band | purpose | minimum accepted trajectories |
| --- | --- | ---: |
| 4–7 mm | visual coarse-direction and overshoot recovery | 8 sectors × 4 = 32 |
| 2.5–4 mm | coarse-to-contact transition | 8 sectors × 6 = 48 |
| 1–2.5 mm | tactile fine alignment under middle/high contact load | 8 sectors × 2 loads × 6 = 96 |
| 0.2–1 mm | rim-oscillation damping and retry | 8 sectors × 2 loads × 8 = 128 |
| <0.2 mm through strict depth | clearance entry and insertion continuation | 8 sectors × 4 = 32 |

Minimum supplement: 336 accepted strict trajectories. If time is constrained, collect the 1–2.5 mm and 0.2–1 mm cells first, but do not call the reduced set formal data.

At least half of the 0.2–1 mm trajectories must begin from actual 5k-policy failure states. Include both:

- excessive lateral-action/overshoot failures;
- downward push while laterally blocked, followed by unload, recenter and retry.

Do not accept trajectories produced by directly teleporting the peg into success. Snapshot restoration is allowed only to reproduce a policy-visited starting state; all recovery motion must be physical controller action.

## Dataset and split contract

- Both arms train on exactly the same trajectory manifest and the same frame order.
- Split by `pair_id`; no source episode or restored state may cross train/validation/test.
- Balance the sampler over `(source, XY band, sector, contact-load band)`.
- Preserve a never-trained same-state evaluation manifest.
- Record source HDF5 SHA-256, Git commit, collector command, policy checkpoint hash, state-snapshot hash and rejection statistics.
- Keep a 10-step strict-success hold and reject pass-through, grasp drift and collision-tail attempts.

For the formal dataset, retain the v4 goal of 320 contact-recovery plus 80 nominal insertions only if the new clearance/rollout cells replace equivalent old cells. Do not simply append hundreds of near-duplicate frames and allow long episodes to dominate.

## Next A100 training sequence

### Gate 0: data-only audit

No VLA training until all of the following hold:

- every required cell has its planned count;
- first expert XY action points toward the hole in at least 95% of 1–4 mm states;
- in the 0.2–1 mm band, expert lateral action magnitude has a documented median and 99th percentile and no collision-tail outliers;
- strict insertion is held for 10 simulation steps;
- a grouped-by-`pair_id` torque probe still beats frozen RGB+proprioception by the v4 Gate-C margins.

### Gate 1: common-base 2k smoke

Train visual and torque-original from the same corrected official base for 2k steps. Use:

```text
seed=1000
batch_size=32
action_loss_first_step_weight=5.0
same optimizer, LR schedule, sampler and checkpoint cadence
```

Save step 2k even if it fails. Run loading, padding, normalization and counterfactual audits. Do not tune one arm independently.

### Gate 2: common-base 10k decision run

Restart both arms from the same official base on the finalized augmented manifest and train to 10k. Save at 2k, 5k, 8k and 10k. This run replaces another 5k-only gate; do not warm-start from the present 5k checkpoints because the data distribution changes.

Every checkpoint must be evaluated in these offline strata:

- 4–7 mm;
- 2.5–4 mm;
- 1–2.5 mm;
- 0.2–1 mm;
- <0.2 mm/insertion continuation.

Minimum 10k go thresholds:

| criterion | visual | torque-original |
| --- | ---: | ---: |
| 4–7 mm first-action cosine | >=0.55 and >=7/8 sectors | >=0.45 and >=7/8 sectors |
| 1–2.5 mm first-action cosine | >=0.45 and >=6/8 sectors | >=0.75 and 8/8 sectors |
| 0.2–1 mm first-action cosine | >=0.45 and >=6/8 sectors | >=0.65 and >=7/8 sectors |
| original vs zero/shuffle | n/a | original better on >=75% of paired states |
| action magnitude | no systematic support violation | no systematic support violation |

The current 5k result already satisfies the torque 1–2.5 mm directional target but fails the <1 mm target.

### Gate 3: same-snapshot closed loop

Use `experiment/eval_factory_peg_insert_same_state_pair.py`. Start with one 2-step smoke pair and require:

- `paired_initial_observation_identical=true`;
- both restore errors at most `1e-6`;
- identical initial RGB and proprioception hashes;
- `common_first_rgb_replayed=true` (one fork RGB is shared; later RGB stays live);
- no ejection or grasp drift.

Then run 8 paired holdout states, followed by at least 32. For the formal result use the frozen 120-state manifest. Report strict insertion, minimum XY error, contact-to-success recovery, ejection, grasp drift and time-to-success.

Independent processes with the same seed are not paired samples. Do not use their difference for a tactile-effect confidence interval.

### Gate 4: formal continuation

Continue beyond 10k only if:

- visual reliably reaches the near-hole region;
- torque-original reaches strict insertion on at least 30% of train-adjacent sanity states;
- original torque beats zero and causal shuffle on the same snapshots;
- failures are no longer dominated by a common initializer, camera or controller problem.

Continue the selected common-base runs to 20k and evaluate every 2k. Stop after three validation evaluations without improvement. Extend toward 50k only when the validation curve is still improving; step count alone is not a success criterion.

## Formal claim rule

Retain the v4 rule: on the same saved 120 states, original torque must exceed both zero and causal-shuffle strict recovery by at least 15 percentage points; both paired bootstrap 95% intervals must exclude zero; gains must cover both load/offset bands and at least 6/8 sectors.

Also report visual-only. A tactile benefit is interpretable only after the shared visual/coarse stage is valid and the same-snapshot audit passes.
