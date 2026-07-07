---
library_name: lerobot
pipeline_tag: robotics
tags:
- smolvla
- vision-language-action
- imitation-learning
- lerobot
- isaaclab
- lstm
base_model: lerobot/smolvla_base
datasets:
- Dleshers/franka-pickplace-joint-visual-torque-w30-v1
inference: false
---

# SmolVLA Franka pick-and-place torque-LSTM — 50K steps, seed 1000

This model is the torque-conditioned arm of a controlled comparison against the visual-only SmolVLA baseline. Both arms use the same official initialization, dataset rows, images, state, actions, seed, batch size, and 50K-step schedule.

## Only experimental change

The model adds `observation.gripper_torque` with shape `[30,1]`. A separately trained causal LSTM (`input=1`, `hidden=32`, `layers=1`, `output=16`) encodes the window. Its weights are loaded from `torque_16d_encoder.pt` and frozen during SmolVLA training.

The 16-dimensional latent passes through a trainable LayerNorm and linear projection to form one Action Expert suffix token. The torque token is not inserted into the VLM prefix. The action output head consumes only the final 50 action-token outputs.

## Training configuration

- Optimizer steps: 50,000
- Seed: 1000
- Batch size: 8
- Optimizer: AdamW
- Peak learning rate: `1e-4`
- Warmup steps: 1,000
- Decay steps: 30,000
- State: 9 dimensions
- Action: 8 dimensions
- Visual observations: two RGB cameras at 224 x 224
- Torque window: `[30,1]`, newest sample at index `-1`
- External LSTM: frozen
- General tactile policy input: disabled

The uploaded directory is the final `pretrained_model` artifact. Intermediate checkpoints and optimizer state are omitted.
