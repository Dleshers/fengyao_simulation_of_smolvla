# Remote Retraining Handoff

## Current decision

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

Apply the two workspace patches to matching LeRobot/IsaacLab clones, install the policy overrides,
then run a two-demo dataset smoke collection before the full collection:

```bash
RAW_DIR=/persistent/path/raw_smoke NUM_DEMOS=2 NUM_ENVS=1 \
  remote_workspace/experiment/rebuild_dataset.sh collect

RAW_DIR=/persistent/path/raw_smoke \
DATASET_REPO_ID=franka_pickplace_joint_visual_torque_w30_smoke \
  remote_workspace/experiment/rebuild_dataset.sh convert
```

The converter has an end-to-end synthetic test proving the 9D/8D schema and causal torque padding.
The real Isaac collector may spend more than one minute retrying remote Nucleus material assets on
first startup; run it in a persistent terminal session.
