# 2026-07-09 next contact-rich experiment plan

## Goal

Move the main tactile-benefit comparison away from visually easy pick-and-place and into a contact-rich task.  The first target task is peg insertion.

The question to answer is no longer whether torque injection can improve pick-and-place.  Pick-and-place has already shown that gated/zero-init torque injection does not collapse a strong visual policy.  The next question is:

> With official SmolVLA pretraining as the starting point, does gated torque information improve or destabilize a contact-sensitive task?

## Official pretraining requirement

Use the official Hugging Face / LeRobot SmolVLA base model:

```text
lerobot/smolvla_base
```

This was verified with `hf models list --search smolvla`, where `lerobot/smolvla_base` is the public LeRobot model tagged for robotics / VLA / SmolVLA.

Do not initialize peg-insert experiments from the local pick-and-place 50k checkpoints. Those checkpoints are downstream pick-and-place artifacts and would confound the new task comparison.

The official pretraining corpus is represented operationally by this official checkpoint.  A separate from-scratch or re-pretraining run on the full official/community LeRobot corpus is out of scope for the local RTX 3060 workflow unless a later cloud/HF Jobs plan is created.

## Available peg-insert task

The local IsaacLab-Tactile workspace contains a manager-based peg insertion task:

```text
Isaac-Peg-Insert-Franka-v0
Isaac-Peg-Insert-Franka-IK-Rel-v0
Isaac-Peg-Insert-Franka-Play-v0
Isaac-Peg-Insert-Franka-IK-Rel-Play-v0
```

Relevant local documentation:

```text
_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/PEG_INSERT_MANAGER_BASED_README.md
```

Recommended control mode for data collection:

```text
Isaac-Peg-Insert-Franka-IK-Rel-v0
```

Reason: it uses relative 6-DoF IK plus gripper command and is the intended teleoperation path.

## Immediate scripts added

1. Restore official SmolVLA base checkpoint:

```bash
bash experiment/download_official_smolvla_base.sh
```

2. Short Isaac headless smoke test for peg insertion:

```bash
bash experiment/peg_insert_env_smoke.sh
```

The smoke script wraps a finite probe:

```text
experiment/peg_insert_headless_probe.py
```

It prints explicit stages:

- `launching_app`
- `app_launched`
- `importing_isaaclab_tasks`
- `imports_done`
- `parsing_env_cfg`
- `making_env`
- reset / step telemetry
- `success`

3. Small raw HDF5 teleoperation collection:

```bash
NUM_DEMOS=10 TELEOP_DEVICE=keyboard bash experiment/collect_peg_insert_hdf5.sh
```

For smoother demonstrations, use:

```bash
NUM_DEMOS=10 TELEOP_DEVICE=spacemouse bash experiment/collect_peg_insert_hdf5.sh
```

## Proposed experiment ladder

### Stage 0: task smoke

Run the zero-action headless smoke test and confirm:

- Isaac starts headless;
- task registers correctly;
- reset succeeds;
- observation/action spaces print;
- one short rollout steps without crash.

No data collection or training should happen until this passes.

### Stage 1: small demonstration feasibility

Collect 5-10 raw HDF5 peg-insert demonstrations.

Audit:

- success flags;
- episode lengths;
- action dimensions;
- observation keys;
- camera availability;
- whether gripper/peg/hole telemetry is present;
- whether gripper force/torque signal is available or must be added.

### Stage 2: LeRobot conversion design

The existing pick-and-place converter/validator assumes:

- state `[9]`;
- action `[8]`;
- two RGB cameras `[3,224,224]`;
- `observation.gripper_torque` window `[30,1]`.

Peg insertion may expose different state/action definitions, especially if using IK-relative actions.  Therefore, do not reuse the pick-and-place converter blindly.  First write a peg-specific conversion audit that reports the exact feature schema.

Required outcome before training:

- fixed LeRobot feature schema;
- explicit action meaning;
- explicit camera keys;
- explicit torque/contact key;
- task instruction string;
- metadata assertions.

### Stage 3: official-pretrain visual baseline

Fine-tune from:

```text
lerobot/smolvla_base
```

on the peg-insert LeRobot dataset with torque disabled.

Purpose:

- establish whether official SmolVLA pretraining can adapt to the simulated peg-insert task;
- create the matched visual baseline.

### Stage 4: gated torque comparison

Fine-tune from the same official base / same dataset with:

```text
policy.use_torque_lstm=true
policy.torque_zero_init_adapter=true
policy.torque_gate_init=1.0
policy.train_torque_lstm=false
```

Use the existing frozen torque encoder only if the peg-insert torque signal has the same semantics as the pick-and-place gripper torque channel.  If the signal differs, train or calibrate a new torque encoder before comparing policies.

### Stage 5: matched evaluation

Evaluate:

- official-pretrain visual baseline;
- official-pretrain gated torque model;
- optional visual-only continuation control.

Use the same seeds, same initial poses, same success logic, and trajectory JSONL logging.

## Current blockers / checks

1. Directly importing `isaaclab_tasks` with the conda Python failed with `ModuleNotFoundError: omni.log`; this is expected before Isaac Sim / Kit is launched.
2. The peg-insert smoke test now uses a finite AppLauncher probe instead of the infinite `zero_agent.py` loop.
3. A peg-insert eval server equivalent to the pick-and-place eval server may need to be added before closed-loop SmolVLA evaluation.
4. A peg-specific HDF5-to-LeRobot converter is likely required; the pick-and-place assumptions must not be copied without schema audit.
5. The torque channel must be confirmed for peg insertion. If only joint torques or contact forces are available, the current gripper-torque LSTM may not be semantically valid.

## Execution status on 2026-07-09

Completed:

- Verified the current official SmolVLA model repo as:

```text
lerobot/smolvla_base
```

- Downloaded the official checkpoint locally:

```text
_runtime/remote_handoff_gripper_lstm_work/pretrained/official_smolvla_base
```

- Restored files include:

```text
model.safetensors
config.json
policy_preprocessor.json
policy_postprocessor.json
policy_preprocessor_step_5_normalizer_processor.safetensors
policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

- Approximate local size: `873M`.

Smoke/probe results:

- A first `zero_agent.py` attempt was unsafe for this purpose because it runs indefinitely. It left a residual Isaac process that locked Omniverse key-value storage; the process was later cleaned up.
- The finite probe then launched Isaac successfully and reached Vulkan/GPU initialization.
- With `ENABLE_CAMERAS=0`, the probe reached:

```text
[PEG_PROBE] app_launched
[PEG_PROBE] importing_isaaclab_tasks
[PEG_PROBE] imports_done
[PEG_PROBE] parsing_env_cfg
[PEG_PROBE] making_env
```

- The probe did not reach reset/step within a 420 second timeout.

Current blocker:

- `gym.make("Isaac-Peg-Insert-Franka-IK-Rel-v0")` stalls during scene creation.
- The peg-insert config references Nucleus-hosted USD assets:

```text
{ISAACLAB_NUCLEUS_DIR}/Factory/factory_peg_8mm.usd
{ISAACLAB_NUCLEUS_DIR}/Factory/factory_hole_8mm.usd
{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd
```

- Local search under `_runtime` found no copies of these USD files.
- Isaac logs also report `OmniHub is inaccessible`.

Interpretation:

The peg-insert Python registration path is valid, and Isaac/Vulkan can start. The immediate blocker is likely remote/Nucleus asset resolution for the peg/hole/table USD files, not SmolVLA or LeRobot training.

Do not start collection, conversion, training, or evaluation until peg-insert assets are restored locally or the task config is changed to use available local assets.

Recommended next fix:

1. Restore the required Factory peg/hole/table USD assets to a local persistent asset directory.
2. Patch peg-insert config to use local `usd_path`s, or configure Isaac's asset root to a working local mirror.
3. Re-run:

```bash
ENABLE_CAMERAS=0 TIMEOUT_SECONDS=420 NUM_STEPS=1 bash experiment/peg_insert_env_smoke.sh
```

4. Only after reset/step succeeds, run camera-enabled smoke:

```bash
ENABLE_CAMERAS=1 TIMEOUT_SECONDS=420 NUM_STEPS=1 bash experiment/peg_insert_env_smoke.sh
```

## Asset restoration details

The three core URLs were probed with HTTP HEAD and returned `200 OK`:

```text
http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/factory_peg_8mm.usd
  Content-Length: 1690754
  ETag: c766c906dc1a70cb7e8a6f0a9457a354

http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/factory_hole_8mm.usd
  Content-Length: 7508864
  ETag: 721aa0238f093b07070cc459ad18dc6a

http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd
  Content-Length: 4584
  ETag: cc5384821e87bdcd44cf20fb461a5363
```

A helper script has been prepared:

```bash
bash experiment/restore_peg_insert_assets.sh
```

It restores these files to:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror
```

The accompanying patch:

```text
patches/2026-07-09_peg_insert_local_asset_root.patch
```

adds support for:

```bash
LOCAL_ISAAC_4_5_ASSET_ROOT=_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror
```

so the peg-insert config can use local USD assets instead of waiting on the remote Nucleus/S3 path during `gym.make`.

Follow-up diagnostic after restoring the three core files:

- `gym.make` still timed out, but the log revealed the concrete unresolved dependency:

```text
Could not open asset @table.usd@ for reference introduced by .../SeattleLabTable/table_instanceable.usd
```

- Therefore `table_instanceable.usd` is not standalone. The restore script now also downloads:

```text
Isaac/Props/Mounts/SeattleLabTable/table.usd
```

- A fallback was also added for setup/smoke runs:

```bash
PEG_INSERT_SIMPLE_TABLE=1
```

This replaces the SeattleLabTable USD with a local procedural cuboid table. The smoke and collection scripts default to this simple table so task registration, reset, and stepping can be validated without depending on the full table asset chain.

## 2026-07-09 smoke resolution

Further testing showed that there were two independent startup blockers:

1. The SeattleLabTable USD was incomplete without `table.usd`.
2. The peg-insert Franka config always created camera sensors and an `rgb_camera` observation group, even when the smoke script was run with `ENABLE_CAMERAS=0`.

Additionally, the Factory peg/hole USD assets still caused slow/hanging `gym.make` behavior even after being wrapped as `RigidObjectCfg` with articulation root disabled.  For pipeline smoke testing, a procedural fallback was added:

```bash
PEG_INSERT_PROCEDURAL_ASSETS=1
PEG_INSERT_SIMPLE_TABLE=1
```

This fallback uses:

- procedural cylinder peg;
- procedural cuboid target/hole placeholder;
- procedural cuboid table.

It is only for environment, camera, teleop, and data-pipeline validation.  It should not be used as the final high-fidelity insertion geometry for experimental conclusions.

Successful no-camera smoke:

```bash
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
ENABLE_CAMERAS=0 PEG_INSERT_DISABLE_CAMERAS=1 \
PEG_INSERT_PROCEDURAL_ASSETS=1 PEG_INSERT_SIMPLE_TABLE=1 \
TIMEOUT_SECONDS=180 DUMP_AFTER_S=60 NUM_STEPS=1 \
bash experiment/peg_insert_env_smoke.sh
```

Observed:

```text
observation_space=Dict('policy': Box(-inf, inf, (1, 49), float32))
action_space=Box(-inf, inf, (1, 7), float32)
[PEG_PROBE] step=0 terminated=[False] truncated=[False]
[PEG_PROBE] success
```

Successful camera-enabled smoke:

```bash
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
ENABLE_CAMERAS=1 PEG_INSERT_DISABLE_CAMERAS=0 \
PEG_INSERT_PROCEDURAL_ASSETS=1 PEG_INSERT_SIMPLE_TABLE=1 \
TIMEOUT_SECONDS=240 DUMP_AFTER_S=90 NUM_STEPS=1 \
bash experiment/peg_insert_env_smoke.sh
```

Observed:

```text
observation_space=Dict(
  'policy': Box(-inf, inf, (1, 49), float32),
  'rgb_camera': Dict('wrist_cam': Box(-inf, inf, (1, 84, 84, 3), float32))
)
action_space=Box(-inf, inf, (1, 7), float32)
[PEG_PROBE] step=0 terminated=[False] truncated=[False]
[PEG_PROBE] success
```

Current implication:

- The manager-based peg-insert task framework can launch, reset, step, and render headlessly.
- The immediate data-pipeline smoke can proceed using procedural fallback geometry.
- Before formal tactile-benefit experiments, the Factory peg/hole USD hang should be resolved or replaced by a high-fidelity procedural/contact geometry that is explicitly documented.

## Stop conditions

Stop before training if any of these fail:

- official `lerobot/smolvla_base` cannot be restored;
- peg-insert task cannot launch headless;
- raw HDF5 collection cannot mark successful demos;
- converted dataset schema is not explicit and validated;
- torque/contact channel is missing or semantically incompatible;
- eval success criterion is ambiguous.

## Next concrete command

First restore the official checkpoint:

```bash
bash experiment/download_official_smolvla_base.sh
```

Then run:

```bash
bash experiment/peg_insert_env_smoke.sh
```
