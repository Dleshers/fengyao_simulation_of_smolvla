# SmolVLA headless simulation handoff

This repository transfers the code and operating notes for evaluating:

1. visual-only SmolVLA;
2. visual SmolVLA with a causal `[30,1]` gripper-torque LSTM token injected into the Action Expert suffix.

Start here:

1. Read [`REMOTE_HEADLESS_EVAL_HANDOFF.md`](REMOTE_HEADLESS_EVAL_HANDOFF.md).
2. Read [`remote_handoff_gripper_lstm/README_FOR_REMOTE_AGENT.md`](remote_handoff_gripper_lstm/README_FOR_REMOTE_AGENT.md).
3. Apply the LeRobot feature-schema patch and supplied policy overrides.
4. Apply only the supplied IsaacLab gripper-torque patch.
5. Download checkpoints from the GitHub Release referenced in the handoff document and verify SHA-256 before evaluation.

Large checkpoints are deliberately stored as Release assets, not in normal Git history.

