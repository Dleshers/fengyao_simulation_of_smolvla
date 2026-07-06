# Remote Agent Handoff: SmolVLA Gripper-Torque LSTM

## Read first

You are configuring two controlled SmolVLA experiments on a remote GPU machine:

1. **Visual-only baseline**: `use_torque_lstm=false`.
2. **Visual + gripper torque LSTM**: `observation.gripper_torque [B,30,1] -> LSTM -> projected hidden token -> Action Expert suffix`.

Read these files before changing code:

```text
instructions/LSTM_ACTION_EXPERT_EXPERIMENT.md
instructions/REMOTE_TRANSFER_MANIFEST_GRIPPER_LSTM.md
```

The primary experiment uses only one-dimensional gripper torque. Full-body joint torque is out of scope.

## Directory contents

```text
remote_handoff_gripper_lstm/
  README_FOR_REMOTE_AGENT.md
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
    REMOTE_CODEX_SETUP_TORQUE_WINDOW_SIM_LEGACY.md
```

## LeRobot installation

Clone the project source described in the setup references. Before copying overrides, record the checked-out commit and make sure the working tree is clean.

Place the supplied files at:

```text
lerobot_overrides/configuration_smolvla.py
  -> src/lerobot/policies/smolvla/configuration_smolvla.py

lerobot_overrides/modeling_smolvla.py
  -> src/lerobot/policies/smolvla/modeling_smolvla.py

lerobot_overrides/test_smolvla_torque_lstm.py
  -> tests/policies/smolvla/test_smolvla_torque_lstm.py
```

Do not overwrite blindly if the remote revision has different APIs. Compare the files and port the same behavior when necessary.

The authoritative torque configuration is:

```text
use_torque_lstm=true
torque_window_key=observation.gripper_torque
torque_window_size=30
torque_input_dim=1
torque_lstm_hidden_dim=32
torque_lstm_output_dim=16
torque_lstm_num_layers=1
torque_lstm_weights_path=/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/trained_lstm_weights/torque_16d_encoder.pt
train_torque_lstm=false
```

## IsaacLab installation

Apply only:

```text
isaaclab_patches/SETUP_torque_window_gripper_IsaacLab.patch
```

First run `git apply --check`. This patch was originally parameterized with a default window length of 32, so all collection and evaluation commands must explicitly set:

```text
--torque-window-size 30
```

Verify the online observation received by LeRobot is exactly:

```text
observation.gripper_torque: float32 [1,30,1]
```

Index `-1` is the newest sample. At episode startup, left-pad by repeating the first valid torque value. Do not zero-pad.

## Files that are references only

The files under `references/` explain the previous pipeline and original LSTM idea. Do not copy `reference_modeling_smolvla.py` over the current implementation: it uses the historical `hidden=64, layers=2` architecture and injects the LSTM result into the VLM prefix. The formal experiment requires `hidden=32, layers=1`, frozen, with an Action Expert suffix token.

`REMOTE_CODEX_SETUP_TORQUE_WINDOW_SIM_LEGACY.md` describes the older flattened linear-window implementation. Use it only for repository and dependency context.

Do not apply any `SETUP_torque_window_*_lerobot.patch`. Do not apply full-body torque patches.

## Required verification

From the LeRobot repository:

```bash
python -m py_compile \
  src/lerobot/policies/smolvla/configuration_smolvla.py \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  tests/policies/smolvla/test_smolvla_torque_lstm.py

pytest -q tests/policies/smolvla/test_smolvla_torque_lstm.py
```

Inspect one dataset batch and assert:

```python
assert batch["observation.gripper_torque"].shape[-2:] == (30, 1)
```

Run 100-500 training steps for both arms before full training. In torque mode, confirm nonzero gradients for `torque_lstm` and `torque_to_expert` and confirm those weights are saved in the checkpoint.

## Fair comparison

Use the same dataset, split, images, state, action representation, initial checkpoint, seed, optimizer, batch size and training steps. The intended independent variable is only:

```text
use_torque_lstm=false  vs  use_torque_lstm=true
```

Finally, run matched full closed-loop IsaacLab rollouts. Compare task success, random motion, action variation, joint velocity/acceleration peaks, collisions, safety-limit violations and rollout videos.
