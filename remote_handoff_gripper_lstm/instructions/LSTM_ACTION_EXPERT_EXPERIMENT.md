# SmolVLA Visual vs Torque-LSTM Experiment

## Experiment definition

Run exactly two controlled experiments in the full robot simulator:

1. **Visual SmolVLA baseline**: standard images, language and robot state. Torque is not used.
2. **Visual + torque-LSTM SmolVLA**: the same inputs plus a causal one-dimensional gripper torque window. An LSTM encodes the window and one projected hidden vector is inserted directly into the Action Expert suffix stream.

The goal is to measure whether temporal torque improves the task and whether it causes unstable or random motion during closed-loop rollout.

Do not use the older `SETUP_torque_window_*` LeRobot patches for this comparison. They flatten the window and apply one linear layer; they are not the LSTM experiment.

## Implemented model

Updated files:

```text
lerobot/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py
lerobot/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

Torque data flow:

```text
observation.gripper_torque [B, 30, 1]
  -> frozen 1-layer LSTM(input=1, hidden=32)
  -> final hidden state
  -> Linear(64, 16)
  -> LayerNorm(16)
  -> Linear(16, action_expert_hidden_size)
  -> one torque suffix token
  -> action-time suffix tokens
  -> Action Expert
```

The torque vector is not placed in the VLM prefix. At inference it is calculated once per policy call and reused during all flow-matching denoising steps. By default, the LSTM and projection are trained end to end.

## Dataset contract

Use one dataset and one fixed train/validation split for both experiments:

```text
observation.images.*
observation.state
action
observation.gripper_torque: [30, 1]
```

Each causal window is `[t-29, ..., t]`. At episode start, repeat the first valid torque value on the left. Do not use zero padding and do not change padding behavior between training and evaluation.

Fit torque normalization statistics on the training split only and reuse them at evaluation time.

## Model switches

Visual baseline:

```text
--policy.use_torque_lstm=false
```

Torque-LSTM:

```text
--policy.use_torque_lstm=true
--policy.torque_window_key=observation.gripper_torque
--policy.torque_window_size=30
--policy.torque_input_dim=1
--policy.torque_lstm_hidden_dim=32
--policy.torque_lstm_output_dim=16
--policy.torque_lstm_num_layers=1
--policy.train_torque_lstm=false
```

Only the one-dimensional gripper signal is part of the current main experiment. Full-body joint torque is out of scope.

## Training

Start both runs from the same SmolVLA checkpoint and keep seed, split, steps, batch size, optimizer and cameras identical.

Baseline:

```bash
lerobot-train \
  --dataset.repo_id=YOUR_DATASET \
  --dataset.root=YOUR_DATASET_ROOT \
  --policy.type=smolvla \
  --policy.pretrained_path=YOUR_SMOLVLA_BASE \
  --policy.use_torque_lstm=false \
  --seed=1000 \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=2000 \
  --output_dir=outputs/smolvla_visual_seed1000
```

Torque-LSTM:

```bash
lerobot-train \
  --dataset.repo_id=YOUR_DATASET \
  --dataset.root=YOUR_DATASET_ROOT \
  --policy.type=smolvla \
  --policy.pretrained_path=YOUR_SMOLVLA_BASE \
  --policy.use_torque_lstm=true \
  --policy.torque_window_key=observation.gripper_torque \
  --policy.torque_window_size=30 \
  --policy.torque_input_dim=1 \
  --policy.torque_lstm_hidden_dim=32 \
  --policy.torque_lstm_output_dim=16 \
  --policy.torque_lstm_num_layers=1 \
  --policy.torque_lstm_weights_path=YOUR_TORQUE_16D_ENCODER_PT \
  --policy.train_torque_lstm=false \
  --seed=1000 \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=2000 \
  --output_dir=outputs/smolvla_torque_lstm_seed1000
```

Check the exact pretrained-path argument with `lerobot-train --help`; it differs across LeRobot revisions.

## Pre-flight validation

```bash
python -m py_compile \
  src/lerobot/policies/smolvla/configuration_smolvla.py \
  src/lerobot/policies/smolvla/modeling_smolvla.py

pytest -q tests/policies/smolvla/test_smolvla_torque_lstm.py
```

Run 100-500 training steps before a long job. Confirm:

- baseline trains without a torque tensor;
- torque mode rejects missing or wrongly shaped input;
- `torque_lstm` and `torque_to_expert` receive nonzero gradients;
- checkpoints contain `torque_lstm.*`, `torque_norm.*`, and `torque_to_expert.*`;
- training and inference use identical torque units, sign, normalization and temporal order.

## Full simulation evaluation

Do not stop at offline loss. Run full closed-loop IsaacLab rollouts for both checkpoints using identical task seeds and episode initializations. The torque server must send `[1,30,1]` under `observation.gripper_torque`, with the latest sample at index `-1`.

Use at least three seeds and record:

- task success and object-drop rates;
- collision count;
- peak joint velocity and acceleration;
- action change `||a_t-a_(t-1)||`;
- torque-window and LSTM-latent norms;
- joint/safety-limit violations;
- rollout video.

If only the torque model moves randomly, first inspect torque units/sign, normalization, sampling frequency, cold-start padding and checkpoint config.

## Fair comparison

The intended independent variable is only:

```text
use_torque_lstm = false  vs  true
```

Do not change cameras, demonstrations, action representation, base checkpoint, seed, training steps or simulator randomization between the two main runs. Frozen-LSTM, full-body torque and alternate-window runs are separate ablations.
