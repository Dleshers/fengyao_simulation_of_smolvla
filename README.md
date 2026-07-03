# SmolVLA headless simulation handoff

This repository transfers the code and operating notes for evaluating:

1. visual-only SmolVLA;
2. visual SmolVLA with a causal `[30,1]` gripper-torque LSTM token injected into the Action Expert suffix.

Start here:

1. Read [`REMOTE_RETRAINING_HANDOFF.md`](REMOTE_RETRAINING_HANDOFF.md) for the current authoritative state.
2. Read [`REMOTE_HEADLESS_EVAL_HANDOFF.md`](REMOTE_HEADLESS_EVAL_HANDOFF.md) for historical evaluation setup.
3. Read [`remote_handoff_gripper_lstm/README_FOR_REMOTE_AGENT.md`](remote_handoff_gripper_lstm/README_FOR_REMOTE_AGENT.md).
4. Apply the current workspace patches and supplied policy overrides.
5. Rebuild the matched dataset and retrain both arms from official `lerobot/smolvla_base`.

Large checkpoints are deliberately stored as Release assets, not in normal Git history.
The old release checkpoints are historical artifacts and are not valid for the final controlled comparison.
