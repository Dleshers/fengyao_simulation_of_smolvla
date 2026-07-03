# Remote Transfer Manifest: Visual vs Gripper-Torque LSTM

## Goal

Configure and run two controlled experiments:

1. Visual-only SmolVLA (`use_torque_lstm=false`).
2. Visual SmolVLA plus `observation.gripper_torque [30,1]`, encoded by an LSTM and injected as one Action Expert suffix token.

## Required files

Transfer these files while preserving their relative destination paths.

### Experiment instructions

```text
simulation/LSTM_ACTION_EXPERT_EXPERIMENT.md
simulation/REMOTE_TRANSFER_MANIFEST_GRIPPER_LSTM.md
```

The remote agent must read `LSTM_ACTION_EXPERT_EXPERIMENT.md` first.

### Modified LeRobot files

Copy into the matching paths of the remote LeRobot clone:

```text
lerobot/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py
lerobot/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
lerobot/lerobot/tests/policies/smolvla/test_smolvla_torque_lstm.py
```

Before replacing files, the remote agent must confirm that its LeRobot revision is compatible and back up or commit its current worktree.

### IsaacLab gripper-torque transport

```text
simulation/SETUP_torque_window_gripper_IsaacLab.patch
```

Apply only the IsaacLab-side gripper patch. Configure its torque window size as `30`, not its older default `32`. Verify that online inference sends:

```text
observation.gripper_torque: [1,30,1]
```

Do not apply `SETUP_torque_window_gripper_lerobot.patch`; it implements a flattened linear projection rather than LSTM.

## Reference files

These explain the original design and remote environment, but are not the authoritative implementation:

```text
smolvla_20260603/smolvla_20260603/force_feedback_interface/readme.md
smolvla_20260603/smolvla_20260603/force_feedback_interface/modeling_smolvla.py
simulation/REMOTE_CODEX_SETUP_TORQUE_WINDOW_SIM.md
simulation/SETUP_TUTORIAL_TACTILE_PIPELINE.md
```

The first two files document the `[30,1]` signal and LSTM concept. The current implementation differs by placing the projected LSTM vector in the Action Expert suffix, not the VLM prefix.

## Optional file

```text
smolvla_20260603/smolvla_20260603/force_feedback_interface/trained_lstm_weights/torque_16d_encoder.pt
```

This is not required for the main end-to-end experiment (`train_torque_lstm=true`). Transfer it only for a separately reported pretrained/frozen-LSTM ablation. Never silently load it in only one of the two primary experiments.

## Files not to use for the main experiment

```text
simulation/SETUP_torque_window_gripper_lerobot.patch
simulation/SETUP_torque_window_fullbody_IsaacLab.patch
simulation/SETUP_torque_window_fullbody_lerobot.patch
```

The first is the obsolete linear encoder. The other two are full-body torque variants, which are outside the current experiment.

## Recommended transfer layout

```text
remote_handoff/
  instructions/
    LSTM_ACTION_EXPERT_EXPERIMENT.md
    REMOTE_TRANSFER_MANIFEST_GRIPPER_LSTM.md
  lerobot_overrides/
    configuration_smolvla.py
    modeling_smolvla.py
    test_smolvla_torque_lstm.py
  isaaclab_patches/
    SETUP_torque_window_gripper_IsaacLab.patch
  references/
    force_feedback_readme.md
    reference_modeling_smolvla.py
    SETUP_TUTORIAL_TACTILE_PIPELINE.md
```

## Instructions for the remote agent

1. Read this manifest and `LSTM_ACTION_EXPERT_EXPERIMENT.md`.
2. Clone the exact IsaacLab-Tactile and LeRobot repositories described by the setup documentation.
3. Inspect the target LeRobot revision before copying the three modified files; merge API differences if necessary.
4. Apply only the gripper IsaacLab patch and set the window length to 30.
5. Verify dataset metadata contains `observation.gripper_torque` with shape `[30,1]`.
6. Run `py_compile` and `test_smolvla_torque_lstm.py`.
7. Run 100-500-step smoke training for both experiment arms.
8. Run matched closed-loop simulation rollouts and record stability metrics and videos.
