# Remote Retraining Handoff

For copy-paste commands to restore the runtime and immediately collect smoke/full HDF5 files, read
`START_HDF5_COLLECTION.md` first.

## Local audit update (2026-07-05)

The repository was audited locally while the remote GPU host was unavailable. The following blocking
issues were fixed:

- evaluation now defaults to `joint` control, matching `state[9]` and `action[8]`; the previous
  `ik_rel` default produced `state[11]` and `action[7]` and was incompatible with the training schema;
- the policy-factory feature refresh patch now removes `observation.gripper_torque` from the visual
  baseline when `use_torque_lstm=false`, so the baseline neither declares nor normalizes torque;
- dataset validation now checks causal overlap between adjacent `[30,1]` windows, episode-boundary
  reset padding, finite values and nonconstant sampled torque;
- IsaacLab collection preflight and raw-HDF5 auditing were added and integrated into dataset rebuild.
- remote setup now installs and preflight-checks `h5py` and `pytest`, which are required by the new
  raw-data audit and unit tests.

New preparation files:

```text
remote_workspace/experiment/preflight_collection.sh
remote_workspace/experiment/inspect_raw_hdf5.py
remote_workspace/experiment/test_validate_dataset_windows.py
```

The local machine can validate source, patches and synthetic data, but it cannot replace the remote
CUDA/Isaac Sim collection, training or closed-loop rollout.

Local verification completed on 2026-07-05:

- Python AST checks passed for the evaluation probe, dataset validator, raw-HDF5 inspector and new
  causal-window tests;
- a dependency-light synthetic dataset passed causal overlap and episode-reset checks, and deliberate
  window corruption was rejected;
- `git diff --check` passed and all three patch files are syntactically parseable;
- real HDF5 execution was not run locally because this Windows Python environment lacks `h5py`;
- Linux `bash -n`, Isaac imports, headless rendering and CUDA checks remain remote preflight tasks.

## Current decision

The authoritative LSTM configuration is `input=1, hidden=32, layers=1, output=16`, loaded from the
verified standalone encoder and frozen during SmolVLA training. Any older `hidden=64`, `layers=2`,
or `train_torque_lstm=true` instruction is obsolete.

The released `visual_050000` and `torque_lstm_030000` checkpoints are incompatible with the final
controlled interface. Keep them only as historical smoke-test artifacts. Do not use them for the
reported visual-vs-torque comparison or as initialization checkpoints.

Both new runs must start from the official `lerobot/smolvla_base` weights and the same rebuilt
dataset. The baseline is original visual SmolVLA. The torque arm adds only a frozen external causal
LSTM token to the Action Expert suffix.

## Authoritative interface

```text
state:  float32 [B,9] = 7 arm joint positions + 2 gripper joint positions
camera1/camera2: [B,3,224,224]
action: float32 [B,8] = 7 absolute arm joint targets + 1 gripper command
torque arm only: observation.gripper_torque float32 [B,30,1]
window order: oldest -> newest; index -1 is newest
episode padding: repeat the first valid sample on the left
external LSTM: input=1, hidden=32, layers=1, output=16; frozen
fusion: one projected token prepended to the Action Expert suffix, never the VLM prefix
```

Read `remote_workspace/experiment/SMOLVLA_TORQUE_INTERFACE.md` for the complete contract.

## Files backed up here

- `remote_handoff_gripper_lstm/lerobot_overrides/`: current SmolVLA implementation and tests.
- `patches/remote_workspace_lerobot.patch`: all required remote LeRobot environment/client changes.
- `patches/remote_workspace_isaaclab.patch`: current IsaacLab server, collector and converter changes.
- `remote_workspace/experiment/`: setup, probes, matched training/evaluation and dataset rebuild scripts.

## Large persistent artifacts not stored in Git

```text
Official base:
/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/remote_handoff_gripper_lstm_workspace/pretrained/smolvla_base_official

Standalone TorchScript LSTM:
/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/trained_lstm_weights/torque_16d_encoder.pt

Dataset destination:
/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/datasets/
```

The LSTM was verified to map `[B,30,1]` to `[B,16]`; loading its weights into the native PyTorch
encoder gives exact output equality (`max_abs_diff=0.0`).

## Resume

Apply the two workspace patches to matching LeRobot/IsaacLab clones, apply the policy-factory patch,
and install the policy overrides. The required LeRobot-side pieces are all three of:

```text
patches/remote_workspace_lerobot.patch
patches/lerobot_factory_refresh_dataset_features.patch
remote_handoff_gripper_lstm/lerobot_overrides/
```

Before collecting data, run the complete preflight (including a minimal Isaac headless launch):

```bash
RUN_ISAAC_SMOKE=1 bash remote_workspace/experiment/preflight_collection.sh
```

Then run a two-demo dataset smoke collection before the full collection:

```bash
RAW_DIR=/persistent/path/raw_smoke NUM_DEMOS=2 NUM_ENVS=1 \
  remote_workspace/experiment/rebuild_dataset.sh collect

RAW_DIR=/persistent/path/raw_smoke \
DATASET_REPO_ID=franka_pickplace_joint_visual_torque_w30_smoke \
  remote_workspace/experiment/rebuild_dataset.sh convert
```

The rebuild script now runs `inspect_raw_hdf5.py` automatically after collection and before
conversion. Run the causal-window unit test in the LeRobot environment as well:

```bash
cd remote_workspace/experiment
../.venv/lerobot/bin/python -m pytest -q test_validate_dataset_windows.py
```

Evaluation commands default to joint mode. Keep `CONTROL_MODE=joint`; using `ik_rel` is a separate,
incompatible interface and must not be used with these checkpoints.

The converter has an end-to-end synthetic test proving the 9D/8D schema and causal torque padding.
The real Isaac collector may spend more than one minute retrying remote Nucleus material assets on
first startup; run it in a persistent terminal session.
