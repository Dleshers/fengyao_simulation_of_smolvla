---
library_name: lerobot
pipeline_tag: robotics
tags:
- smolvla
- vision-language-action
- imitation-learning
- lerobot
- isaaclab
base_model: lerobot/smolvla_base
datasets:
- Dleshers/franka-pickplace-joint-visual-torque-w30-v1
inference: false
---

# SmolVLA Franka pick-and-place baseline — 50K steps, seed 1000

This is the visual-only control model for a matched comparison with a gripper-torque-LSTM SmolVLA policy. It is fine-tuned from `lerobot/smolvla_base` on 200 synthetic Isaac Lab / TacEx demonstrations.

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
- Torque LSTM: disabled
- Tactile policy input: disabled

The uploaded directory is the final `pretrained_model` artifact needed for inference. Optimizer state and intermediate checkpoints are intentionally omitted.

## Intended use

Use this checkpoint as the visual-only baseline for evaluation in the matching Franka pick-and-place simulation. It has not yet been validated for real-robot deployment or safety-critical use.

Training and evaluation scripts are tracked in `Dleshers/fengyao_simulation_of_smolvla`.
