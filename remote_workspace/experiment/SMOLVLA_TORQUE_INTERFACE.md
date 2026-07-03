# SmolVLA Gripper-Torque Experiment Contract

## Controlled comparison

The baseline is the original visual SmolVLA policy. Both arms use exactly the same:

- `observation.state`: Franka absolute joint state, float32 `[B,9]` (7 arm + 2 gripper joints)
- `observation.images.camera1`: float32 `[B,3,224,224]`
- `observation.images.camera2`: float32 `[B,3,224,224]`
- language task, action normalization, initial SmolVLA checkpoint, split, seed and training schedule
- `action`: float32 `[B,8]`, containing 7 absolute arm-joint targets and one gripper command

The baseline must not declare, normalize or consume a torque feature.

## Torque arm

The torque arm adds exactly one input:

```text
observation.gripper_torque: float32 [B,30,1]
```

The last index is the newest sample. At episode boundaries, left-pad by repeating the first valid
sample. Never mix windows across episodes and never zero-pad.

A separately trained causal LSTM (`input=1`, `hidden=32`, `layers=1`, `output=16`) maps
`[B,30,1]` to `[B,16]`. Load it from
`torque_lstm_weights_path` and freeze it (`train_torque_lstm=false`). Apply LayerNorm and a trainable
linear projection from 16 dimensions to the Action Expert hidden width, producing one token
`[B,1,D_expert]`. Prepend that token to the Action Expert suffix immediately before the 50 noisy
action-time tokens. Do not insert torque into the VLM prefix and do not modify the visual/state path.
The Action Expert output head must consume only the final 50 action-token outputs, excluding the
torque token.

## Required checks

1. Verify both arms use the same dataset rows and shared feature tensors.
2. Verify action semantics are absolute joint targets, not deltas or end-effector commands.
3. Verify camera identity/order and preprocessing match collection.
4. Verify the standalone LSTM architecture and weights load strictly.
5. Verify LSTM parameters are frozen while `torque_norm` and `torque_to_expert` receive gradients.
6. Run a short overfit/smoke test and log predicted action ranges before closed-loop evaluation.

## Message for the dataset-machine agent

Please audit the existing LeRobot dataset against this exact controlled experiment contract. The
baseline must be original visual SmolVLA with shared inputs `observation.state [9]`, camera1 and
camera2 `[3,224,224]`, and `action [8] = seven absolute Franka joint targets plus one gripper
command`; it must not consume torque. The tactile arm must use the identical rows, images, state,
actions, split and normalization, adding only `observation.gripper_torque float32 [30,1]`, where
index `-1` is newest, windows never cross episode boundaries, and startup is left-padded by repeating
the first valid value. Please report the actual feature names/shapes/dtypes, camera identities and
order, collection frequency, state/action semantics, per-dimension state/action statistics, torque
statistics, and checks for NaN/Inf or constant channels. Also identify the standalone causal LSTM
checkpoint, its input/hidden/output dimensions and layer count, verify it converts `[B,30,1]` to
`[B,16]`, and confirm whether its weights were intended to remain frozen. Flag any discrepancy such
as 6D state, three cameras, 256x256 images, delta/EEF actions, a different task scene, or incompatible
torque units before any retraining or evaluation.
