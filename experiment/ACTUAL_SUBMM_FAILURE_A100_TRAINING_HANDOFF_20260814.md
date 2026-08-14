# Actual-policy sub-mm recovery: A100 training handoff (2026-08-14)

## Decision and scope

This is the latest instruction after the corrected 5k current-action-loss gate. The RTX collector supplements the audited balanced64 corpus with 16 difficult trajectories that were actually visited by the frozen 5k hybrid policy. It does not claim a formal tactile result by itself.

The 16 new episodes are balanced over 8 approach sectors and 2 contact-load bands. Visual performs coarse alignment, torque-original latches at 3.5 mm, and a trajectory is retained only when the closed loop reaches a physically blocked state below 1 mm. Oracle pose is audit/label-generation metadata only and is never an observation.

## Authoritative artifacts

- Raw, audits, combined HDF5 and manifest: [Dleshers/factory-peg-insert-policy-failure-recovery-v1-hard16](https://huggingface.co/datasets/Dleshers/factory-peg-insert-policy-failure-recovery-v1-hard16)
- Ready-to-train LeRobot 30x7 dataset: [Dleshers/factory-peg-insert-contact-recovery-v2-hard80-lerobot](https://huggingface.co/datasets/Dleshers/factory-peg-insert-contact-recovery-v2-hard80-lerobot)
- Collector: `experiment/collect_factory_peg_insert_policy_failure_recovery_v1.py`
- Admission audit: `experiment/audit_factory_peg_insert_policy_failure_recovery_v1.py`
- Deterministic merger: `experiment/materialize_factory_peg_insert_contact_recovery_v2.py`
- End-to-end producer: `experiment/run_actual_submm_failure_data_pipeline_20260814.sh`

Do not train until both Hub repositories contain `completion.json`, the hard16 audit says `all_valid=true`, and the manifest hashes match the downloaded files.

## Data contract

The shared corpus contains 80 trajectories:

- 64 previously audited native-contact recoveries, balanced 8 per sector;
- 16 actual-policy sub-mm failures, exactly one per `(sector, load_band)`.

Every hard episode must satisfy all of the following:

- native Factory reset and controller-only physical contact;
- 30 chronological signed 7D joint-torque frames;
- no pose write or state teleport;
- frozen visual-to-torque policy rollout;
- measured failure XY below 1 mm while depth remains in the blocked rim band;
- only oracle recovery frames are behavior-cloning labels;
- pre-action frame alignment and at least 10 strict-success hold frames.

The converter uses:

```bash
--torque-control original --torque-dim 7 --policy-label-only --policy-phase-min 5
```

Visual and torque arms must train from this identical Hub revision, frame order and sampler. The visual arm may ignore the torque feature, but it must not use a separately filtered dataset.

## A100 preflight

```bash
REPO=/path/to/fengyao_simulation_of_smolvla
LEROBOT_ROOT=/path/to/lerobot-tactile
PY=/path/to/python

git -C "$REPO" pull --ff-only
PYTHONPATH="$LEROBOT_ROOT/src" "$PY" -m pytest -q   "$REPO/remote_handoff_gripper_lstm/lerobot_overrides/test_smolvla_torque_lstm.py"

hf download Dleshers/factory-peg-insert-policy-failure-recovery-v1-hard16   --repo-type dataset --local-dir "$PWD/raw_hard16"
hf download Dleshers/factory-peg-insert-contact-recovery-v2-hard80-lerobot   --repo-type dataset --local-dir "$PWD/factory-peg-insert-contact-recovery-v2-hard80-lerobot"

test -f "$PWD/raw_hard16/completion.json"
test -f "$PWD/factory-peg-insert-contact-recovery-v2-hard80-lerobot/completion.json"
```

Also verify that the installed LeRobot SmolVLA override contains the corrected `action_is_pad` current-action loss and `action_loss_first_step_weight`. The legacy 1D `trained_lstm_weights/torque_16d_encoder.pt` is incompatible and must not be loaded.

## Common-base training sequence

Do not resume the diagnostic 5k checkpoints. Start both arms from the same padding-fixed official SmolVLA base.

Common settings:

```text
dataset.repo_id=Dleshers/factory-peg-insert-contact-recovery-v2-hard80-lerobot
seed=1000
batch_size=32
action_loss_first_step_weight=5.0
same base, optimizer, learning-rate schedule, sampler, frame order and checkpoint cadence
```

Only model difference:

```text
visual:
  use_torque_lstm=false

torque-original:
  use_torque_lstm=true
  torque_window_key=observation.gripper_torque
  torque_window_size=30
  torque_input_dim=7
  torque_lstm_hidden_dim=32
  torque_lstm_output_dim=16
  torque_lstm_num_layers=1
  train_torque_lstm=true
  torque_lstm_weights_path=""
```

Run a common-base 2k smoke first. Save both checkpoints and audit offline actions on the same episodes. If loading, normalization, padding, action scale, or visual coarse direction is wrong in either arm, stop and fix the common issue rather than expanding data.

If the 2k gate is healthy, restart both arms from the same official base and train to 10k, saving 2k/5k/8k/10k. Do not warm-start the 10k decision run from the smoke.

## Required evaluation

Use `n_action_steps=1`. Evaluate:

1. offline first-action direction/magnitude in ordinary contact and hard16 strata;
2. same-snapshot paired visual versus torque-original;
3. torque-original versus zero-torque and causal-shuffle counterfactuals;
4. strict insertion, near-hole recovery, ejection/grasp drift and time-to-success by sector/load.

Phase-5 online evaluation retains `--pre-takeover-unload-steps 15`. Evaluation must use the same visual coarse stage and the same 3.5 mm latched transition for all torque counterfactuals.

A tactile claim requires original torque to beat both zero and causal shuffle on the same saved states, with gains covering both load bands and at least 6/8 sectors. A visual/coarse common failure invalidates the comparison.

## Go/no-go

- 2k is only an implementation smoke.
- At 10k, continue to 20k only if visual reliably reaches the near-hole region and torque-original shows causal improvement over both controls.
- Stop after three validation checkpoints without improvement.
- Do not call hard80 a formal 320+80 dataset. It is a time-constrained, targeted decision dataset intended to determine whether a larger formal collection is justified.
