# 2026-07-09 Torque Injection Compatibility Audit

This note records the cautious compatibility checks before further tactile/torque validation.

## Scope

Checked whether the current gated/zero-init gripper-torque LSTM injection is compatible with the existing targeted5 pick-place SmolVLA policy and whether it can be directly transferred to the new peg-insert task.

No full training or batch closed-loop evaluation was started in this audit.

## Code/config facts

Current LeRobot-Tactile SmolVLA implementation uses the intended torque encoder architecture:

```text
input_dim=1
hidden_dim=32
num_layers=1
output_dim=16
train_torque_lstm=false
```

The torque branch is injected as one Action Expert suffix token:

```text
[B,30,1] -> frozen LSTM -> [B,16] -> LayerNorm -> Linear -> [B,1,D_expert]
```

The model then slices action outputs with:

```python
suffix_out = suffix_out[:, -self.config.chunk_size :]
```

Therefore the added torque token is not consumed by the action output head as an extra action token.  It can condition the action tokens through attention, but it does not change action sequence length or action dimensionality.

## Strict LSTM checkpoint verification

Command:

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
  remote_workspace/experiment/verify_frozen_torque_encoder.py \
  trained_lstm_weights/torque_16d_encoder.pt
```

Result:

```text
OK: strict external encoder load: input=1 hidden=32 layers=1 output=16
OK: encoder frozen; LayerNorm and Action Expert projection receive gradients
OK: [B,30,1] -> [B,16] -> [B,1,D_expert]
```

## Unit tests

The first pytest run was polluted by system ROS pytest plugins and failed before collecting LeRobot tests because `launch_testing` required `lark`.

Rerun with plugin autoload disabled:

```bash
cd _runtime/remote_handoff_gripper_lstm_work/lerobot-tactile
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.venv/lerobot/bin/python \
  -m pytest tests/policies/smolvla/test_smolvla_torque_lstm.py -q
```

Result:

```text
3 passed in 0.69s
```

## Dataset and checkpoint compatibility audit

A read-only audit script was added:

```text
experiment/audit_torque_injection_compatibility.py
```

For the current targeted5 pick-place comparison:

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
  experiment/audit_torque_injection_compatibility.py \
  --dataset-root _runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean \
  --visual-policy _runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model \
  --torque-policy _runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_seed1000_20260709/checkpoints/005000/pretrained_model \
  --task-mode pick_place_joint
```

Key result:

```text
PASS: dataset observation.state shape is (9,)
PASS: dataset action shape is (8,)
PASS: dataset camera1 shape is (3, 224, 224)
PASS: dataset camera2 shape is (3, 224, 224)
PASS: dataset gripper torque shape is (30, 1)
PASS: visual baseline does not declare torque input
PASS: action output feature matches between visual and torque policies
PASS: torque_window_size=30
PASS: torque_input_dim=1
PASS: torque_lstm_hidden_dim=32
PASS: torque_lstm_output_dim=16
PASS: torque_lstm_num_layers=1
PASS: train_torque_lstm=false
PASS: torque_zero_init_adapter=true
PASS: torque_gate_init=1.0
INFO: learned torque_gate=0.988035
PASS: pick_place_joint schema matches current SmolVLA checkpoints
SUMMARY: WARN
```

The warning is important:

```text
WARN: torque window is normalized before the frozen LSTM; this is train/eval-consistent,
but only physically correct if the external LSTM was intended to consume normalized torque
```

Raw torque stats from the dataset:

```text
mean=-24.9064
std=21.1363
q50=-40.2717
q99=0.00838649
```

Interpretation:

- The current trained/evaluated torque policy is internally consistent because training and eval both use the same policy preprocessor.
- However, the frozen external LSTM currently receives normalized torque, because `observation.gripper_torque` is registered as feature type `STATE` and the normalizer maps `STATE -> MEAN_STD`.
- If the standalone LSTM was trained on raw torque units, this reduces the physical meaning of the frozen encoder.  This is not a hard runtime incompatibility, but it should be controlled in the next experiment.

## Peg-insert compatibility check

The same audit in `peg_insert_ik` mode reports:

```text
WARN: peg_insert_ik env smoke reports policy state [49] and action [7];
this dataset/checkpoint reports state (9,), action (8).
Do not reuse pick-place checkpoint/converter directly.
```

Therefore, the current pick-place SmolVLA checkpoints and dataset converter are not directly compatible with peg-insert.  Peg-insert needs a dedicated schema/converter/model config.

Current peg-insert smoke schema:

```text
policy state: [49]
action: [7] = arm_action[6] + gripper_action[1]
rgb_camera.wrist_cam: [84,84,3] uint8
```

Current pick-place SmolVLA schema:

```text
observation.state: [9]
action: [8]
camera1: [3,224,224]
camera2: [3,224,224]
observation.gripper_torque: [30,1]
```

## Minimal policy forward check

Loaded both local policies and ran one raw dataset sample through their saved preprocessors and postprocessors.

Policies:

- visual: `targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model`
- gated torque: `targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_seed1000_20260709/checkpoints/005000/pretrained_model`

Result:

```text
visual50k: raw_action_shape=(1, 8) post_shape=(1, 8) finite=True
gated_torque5k: raw_action_shape=(1, 8) post_shape=(1, 8) finite=True
diff: l2=0.0172722 linf=0.0120466 gripper_delta=-0.00365043
```

Interpretation:

- The torque-injected policy produces finite actions with the same output shape as the visual baseline.
- On this sample, adding the trained gated torque branch does not produce a large action jump.

## Torque perturbation sensitivity check

Using the gated torque policy on the same sample, the torque window was replaced with several controlled variants:

```text
real
zeros
dataset_mean
strong_contact_q50
positive_q99
```

Largest observed delta vs the real torque window:

```text
linf <= 0.0168
gripper_delta <= 0.0145
```

Interpretation:

- The learned gated torque branch is not explosively sensitive to reasonable torque-window perturbations.
- This supports the narrower claim that the current injection path does not immediately destabilize the policy.
- It does not prove tactile benefit; pick-place is visually easy and weakly tactile-dependent.

## Current conclusion

For the targeted5 pick-place policy:

- task/robot state/action schema is compatible;
- visual baseline and torque policy share state, cameras, action feature, preprocessing, and postprocessing;
- torque LSTM architecture and weights match the intended authoritative configuration;
- the encoder is frozen and the adapter/gate are trainable;
- output shape and single-sample action range are sane;
- torque perturbations do not cause large action jumps.

The main caveat is torque normalization before the frozen LSTM.  This is not a blocker for evaluating the already-trained gated policy, but it should be made explicit in future controlled experiments:

1. either train/use the standalone torque LSTM on normalized torque windows;
2. or mark torque as an unnormalized/identity feature before the frozen LSTM;
3. or add an explicit raw-torque bypass before policy normalization.

For peg-insert:

- do not reuse the pick-place dataset, converter, or checkpoint directly;
- first define a peg-insert-specific LeRobot schema with `[49]` state, `[7]` action, wrist camera handling, and a clearly specified torque/contact feature;
- then repeat this compatibility audit before any training.

## Suggested next command

For current pick-place gated torque validation, proceed with a small closed-loop evaluation only after confirming the eval server is running:

```bash
PORT=5562 bash experiment/eval_gated_torque_n10.sh client
```

For peg-insert, the next safe step is not training.  It is a schema/converter dry run for a tiny raw collection.

## 1-episode closed-loop compatibility eval

After the offline audits passed, a minimal gated-torque closed-loop run was performed.

Server:

```bash
PORT=5562 EVAL_NAME=targeted5_gated_torque_compat_1ep_seed1000_20260709 \
  bash experiment/eval_gated_torque_n10.sh server
```

Client:

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/lerobot-eval \
  --policy.path="_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_seed1000_20260709/checkpoints/005000/pretrained_model" \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost \
  --env.server_port=5562 \
  --env.task=pick_place \
  --env.observation_height=224 \
  --env.observation_width=224 \
  --env.control_mode=joint \
  --env.torque_window_size=30 \
  --env.include_gripper_torque_window=true \
  --rename_map='{"observation.images.rgb_table":"observation.images.camera1","observation.images.rgb_wrist":"observation.images.camera2"}' \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --seed=1000 \
  --output_dir="_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_gated_torque_compat_1ep_seed1000_20260709"
```

Result:

```text
n_episodes: 1
pc_success: 100.0
eval_s: 40.296
trajectory rows: 200
video: _runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_gated_torque_compat_1ep_seed1000_20260709/videos/isaaclab_tactile_remote_0/eval_episode_0.mp4
```

Trajectory notes:

- success occurred at step 199;
- `success_source=termination_manager`;
- `termination_terms.success=True`;
- `success_term_disagrees_with_reported=False`;
- `post_step_scene_was_auto_reset=True`, so the final `_after` pose in the last JSONL row is already reset-state and should not be interpreted as the final success pose.

Selected pre-success telemetry:

```text
step 100:
  eef_cube_dist_after=0.03336
  cube_pos_after=[0.38033, -0.09622, 0.05505]
  gripper command sign=-1

step 150:
  eef_cube_dist_after=0.03336
  cube_pos_after=[0.47590, 0.04173, 0.20683]

step 198:
  cube_pos_before=[0.53776, 0.12574, 0.07646]
  basket_pos_before=[0.53936, 0.12504, -0.00300]
  eef_cube_dist_before=0.03336

step 199:
  is_success=True
  done=True
```

Server/runtime caveats:

- Isaac startup for the pick-place/basket task was slow: reaching `Server listening` took about 140-160 seconds.
- The basket material path still references missing remote SimReady/Groot texture assets:

```text
omniverse://simready.ov.nvidia.com/Projects/Groot_Content/Material Library/Additional Textures/Fingers_A_rough.png
```

This produced renderer material errors but did not prevent the successful rollout.

- Stopping the server with Ctrl-C after eval completion triggered a non-fatal Isaac cleanup abort:

```text
pybind11::error_already_set: ReferenceError: weakly-referenced object no longer exists
```

This happened during tiled camera / replicator cleanup after the client had already written `eval_info.json`, video, and trajectory logs.

Updated conclusion after closed-loop check:

- The current gated/zero-init torque injection path is operational in closed-loop pick-place.
- It did not break task execution in this 1-episode smoke eval.
- This supports the claim that the torque suffix-token injection can be made non-disruptive.
- It still does not prove tactile benefit; a proper comparison needs matched multi-seed visual vs gated-torque vs torque-ablated evaluations.
