# Baseline local offline inference audit

This is a local, non-closed-loop inference audit. It does not run Isaac evaluation.

- Samples: 600 frames = 3 frames per episode at 20%, 50%, and 80% progress.
- Policy: `/media/ubuntu2204/3342f61c-c19e-4310-b5e6-8d99771c5500/isaac_workspace/fengyao_simulation_of_smolvla/_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- Dataset: `/media/ubuntu2204/3342f61c-c19e-4310-b5e6-8d99771c5500/isaac_workspace/fengyao_simulation_of_smolvla/_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1`

Key metrics:

- Mean batch forward loss: `0.022345`
- MAE by action dim: `[0.010629680007696152, 0.022187339141964912, 0.013652070425450802, 0.009936160407960415, 0.01641594059765339, 0.020655520260334015, 0.02683768980205059, 0.009035740047693253]`
- Arm L2 mean / p90 / p99: `0.056118` / `0.102452` / `0.294564`
- Gripper sign accuracy: `1.0000`
- Gripper MAE: `0.009036`

Files:

- `summary.json`: aggregate metrics and worst samples.
- `samples.csv`: per-sample target/pred/raw action, errors, gripper sign, episode/frame metadata.
