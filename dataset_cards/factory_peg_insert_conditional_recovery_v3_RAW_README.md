---
task_categories:
- robotics
tags:
- isaac-sim
- peg-in-hole
- multimodal
- force-torque
license: other
---

# Factory peg insert: conditional recovery v3 raw data

This private research dataset contains the completed raw HDF5 output of the
conditional near-rim recovery experiment. It is intended for transfer to an
authorized A100 training machine, not as a public benchmark.

Each demonstration has aligned pre-action RGB observations, proprioception,
signed 7D joint torque, a 6D delta-pose action, and recovery metadata. The
recovery starts are deterministic hand-IK plus grasped-peg state
interventions; they are not calibrated physical force impulses.

See the accompanying GitHub repository and `docs/A100_TRAINING_FROM_HF.md` for
the data contract, conversion, training controls, and evaluation procedure.
