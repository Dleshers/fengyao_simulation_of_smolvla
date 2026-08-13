# Contact-recovery v4 dataset design

## Decision objective

The experiment must answer a narrow causal question: after vision has brought
the peg to the hole and rim contact has begun, does signed 7D joint-torque
history improve lateral recovery and strict insertion?

This dataset is not intended to teach coarse hole search.  The previous v3
dataset mixed above-rim geometric recovery with contact recovery and therefore
did not force the policy to use torque.

## Non-negotiable causal structure

1. Use a vision/oracle coarse stage to reach a common pre-contact pose.
2. Re-run the same base seed with balanced lateral directions.  A `pair_id`
   identifies trials sharing the hole pose, camera frame, robot reset, offset
   magnitude and contact-load target.
3. Establish contact slowly.  The contact intervention is never stored as a
   demonstrated policy action.
4. Freeze RGB at the accepted contact state for the causal-training view while
   retaining live RGB in the raw HDF5 for a later deployment view.
5. Record a real 30-step pre-action torque history; do not left-pad the first
   correction frame by repeating one torque sample.
6. Store only corrective oracle actions after the state passes all acceptance
   checks.
7. Split train/validation/test by `pair_id`, never by frame or trajectory.

The model inputs remain RGB, 12D proprioception and optionally a `(30, 7)`
signed joint-torque window.  Peg/hole ground-truth pose and contact audit
signals are metadata only and must never enter the policy input.

## State grid

The exact offset and load thresholds are calibrated in the preflight stage;
they must not be assumed from commanded displacements.  Initial candidate
ranges are:

- direction: 8 balanced sectors (`+x`, `+x+y`, `+y`, `-x+y`, `-x`, `-x-y`,
  `-y`, `+x-y`);
- actual post-contact lateral offset: two bands, approximately
  `[0.6, 1.0] mm` and `(1.0, 1.6] mm`;
- contact load: two empirical bands derived from calibration torque-excursion
  quantiles (middle and high, excluding the upper 10% collision tail);
- fixed-asset XY and yaw: randomized independently of the robot's absolute
  hand pose so proprioception alone cannot reveal relative error;
- approach height and controller phase: narrowly randomized within the stable
  contact-acquisition range.

Offsets are measured after physics settles.  Commanded offset is never used to
assign a stratum.

## Recovery oracle

After contact acceptance the label controller performs three explicit phases:

1. unload: if contact load is rising, retreat vertically by a small bounded
   increment without changing XY;
2. recenter: hold Z and move toward the known hole centre in bounded XY
   increments; do not descend while lateral error is outside the calibrated
   insertion-clearance band;
3. insert: descend slowly only after recentering, and return to unload/recenter
   if load rises again.

This produces repeated touch-correct-retry examples instead of one direct
ground-truth jump.  Phase labels are audit metadata, not policy inputs.

## Raw record contract

Each trajectory stores pre-action-aligned arrays:

- `state`: 12D joint position plus fingertip position;
- `rgb_live_table`, `rgb_live_side`;
- `rgb_contact_frozen_table`, `rgb_contact_frozen_side`;
- `joint_torque`: signed 7D values;
- `applied_wrench`: 6D controller/contact proxy;
- `action`: 6D Factory delta pose;
- `phase`: approach, contact-history, unload, recenter or insert;
- `is_policy_label`: false for approach/intervention history and true only for
  corrective actions.

Trajectory attributes include `pair_id`, `base_seed`, measured offset vector,
direction sector, contact-load band, contact torque baseline/excursion,
pre-contact and post-contact pose, maximum grasp drift, success hold count and
all rejection reasons.

## Acceptance gates before a trajectory is saved

All conditions are required:

- accepted actual XY offset lies in its requested calibrated band;
- state is not already a strict success;
- torque excursion exceeds the calibrated contact threshold and remains below
  the collision-tail threshold;
- contact acquisition changes XY by at most `0.25 mm` and grasp offset by at
  most `2 mm`;
- at least 30 genuine chronological torque samples exist before or through the
  first corrective action;
- the first recenter actions have positive dot product with the ground-truth
  centre direction (mean normalized dot product at least `0.70`);
- final strict insertion is held for 10 consecutive simulation steps;
- no pass-through below the calibrated insertion-depth band occurs.

Rejected attempts remain in an audit log but never enter the training split.

## Stop-go sequence

### Gate A: geometry and contact calibration

Collect attempts only, without training.  Estimate actual offset response,
torque-excursion quantiles, collision/ejection rate and the stable vertical
approach increment.  Proceed only if at least 80% of accepted contact attempts
stay in-band without lateral ejection.

### Gate B: 64-trajectory paired smoke set

Collect 8 directions x 2 offset bands x 2 load bands x 2 repetitions.  Require:

- every cell contains exactly two accepted successful trajectories;
- at least 95% strict recovery success before rejection filtering would make
  the collection impractically selective;
- no direction or load cell differs in count;
- frame/action alignment and all tensor shapes pass the existing dataset audit.

### Gate C: pre-training information audit

Cross-validation is grouped by `pair_id`.  Use simple fixed-capacity probes,
not the VLA:

- an 8-way classifier from the 30x7 torque history must achieve macro-F1 at
  least `0.55` (chance is 0.125);
- the same classifier using frozen RGB plus proprioception must remain at or
  below `0.30` macro-F1;
- adding torque must improve macro-F1 by at least `0.20`;
- a regression probe using torque must reduce corrective-action XY angular
  error by at least 15 degrees relative to frozen RGB plus proprioception;
- label-shuffled torque must collapse toward chance.

Failure means the contact generator or observation design is revised.  Full
collection and VLA training are prohibited.

### Gate D: short overfit/plumbing test

Fine-tune only the torque arm on the smoke set for 2,000 steps.  On held-out
paired smoke states require:

- original torque changes the predicted XY correction relative to zero and
  causal-shuffle inputs;
- the original-torque correction has a positive centre-direction dot product
  on at least 75% of states;
- a small closed-loop run shows no systematic ejection or grasp drift.

This gate detects converter, normalization, LSTM-loading and inference wiring
errors before expensive training.

### Gate E: formal data

Only after Gates A-D pass, collect:

- 320 contact-recovery trajectories: 8 directions x 2 offset bands x 2 load
  bands x 10 repetitions;
- 80 nominal strict insertions to preserve the base task;
- a separate, never-trained 120-state paired evaluation manifest, balanced by
  direction, offset and load.

Recovery trajectories may be weighted by trajectory, but not duplicated by
frame in a way that lets long hard episodes dominate.  Use a balanced sampler
over `(nominal/recovery, direction, offset, load)` cells.

## Time-limited training plan

1. Initialize visual and torque-original independently from the same padding-fixed official SmolVLA base; do not warm-start from formal400 or other old checkpoints.
2. Fine-tune the visual and original-torque arms first; do not immediately run
   four independent 50k trainings.
3. Evaluate the torque arm with original, zero and causal-shuffle torque on the
   exact same saved state manifest.  Zero and shuffle are inference
   interventions, not replacements for the visual baseline.
4. Train additional zero/shuffle arms only if the first causal result is
   positive and a same-capacity training control is needed for publication.

Checkpoint selection uses validation strict recovery, not training loss.
Evaluate every 2,000 steps and stop when validation has failed to improve for
three evaluations.

## Formal conclusion rule

The primary endpoint is paired strict recovery after accepted contact.  Also
report alignment recovery, ejection, grasp drift and time-to-success.  Use the
same 120 saved initial states for every arm/input intervention, paired
bootstrap confidence intervals and McNemar's test.

A tactile benefit is claimed only when:

- original torque exceeds both zero and causal-shuffle strict recovery by at
  least 15 percentage points;
- the 95% paired bootstrap interval for both differences excludes zero;
- the benefit appears in both offset bands and at least six of eight direction
  sectors;
- visual-only remains a valid coarse-alignment baseline rather than failing
  because of a broken initializer.

If original torque itself recovers fewer than 30% of valid train-adjacent
sanity states, stop and revise the recovery oracle/data distribution instead of
scaling evaluation.
