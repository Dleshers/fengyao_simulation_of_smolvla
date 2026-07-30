# Conditional recovery v3: 7D torque formal experiment status

## Purpose

This is the follow-up experiment to the preliminary causal-recovery benchmark.
It tests whether a full signed 7D joint-torque history improves recovery from a
measured near-rim peg-in-hole error over visual-only, zero-torque, and
episode-shuffled-torque controls.

## Data contract

- Raw format: `factory_peg_insert_conditional_recovery_v3` HDF5.
- 100 nominal strict insertions plus 100 strict recovery successes per stratum:
  easy `[2.5, 4.5) mm`, medium `[4.5, 6.0) mm`, hard `[6.0, 7.5) mm` initial XY error.
- A recovery initial state is a deterministic hand-IK plus grasped-peg state
  intervention. It is *not* a calibrated contact-force impulse.
- The intervention action is not labelled. Stored frames begin at the measured
  non-success state and contain only corrective oracle actions.
- `joint_torque` is retained as signed 7D input. The converter emits a causal
  `(30, 7)` history rather than the historical scalar torque norm.
- Recovery episodes are materialized twice during conversion, so recovery
  corrective frames receive higher sampling mass without altering labels.

## Controls and automated sequence

`experiment/run_factory_conditional_recovery_v3_pipeline.sh` performs, in order:

1. collect/resume the 400 raw demonstrations and audit their strict success,
   strata, frame alignment and recovery labels;
2. create original, zero and episode-shuffled 7D LeRobot datasets;
3. train visual, original-torque, zero-torque and shuffled-torque arms for
   50,000 steps each, sequentially;
4. run matched 100-episode (34/33/33 strata) deterministic state-recovery
   evaluation for every arm and write `REPORT.md`.

The 7D torque LSTM is trained from scratch. The legacy
`trained_lstm_weights/torque_16d_encoder.pt` has one input channel and must not
be loaded for this 7D experiment.

## GitHub versus artifact transport

GitHub stores the scripts and this experimental contract only. Runtime folders,
HDF5, converted datasets, pretrained models, checkpoints and logs remain
ignored. For an A100 training machine, transfer the completed raw HDF5
directory (expected roughly 0.7--1.0 GiB) and the official SmolVLA pretrained
directory (about 0.9 GiB), then reproduce conversion/training with this commit.
Do not use ordinary GitHub Git objects for those generated artifacts.
