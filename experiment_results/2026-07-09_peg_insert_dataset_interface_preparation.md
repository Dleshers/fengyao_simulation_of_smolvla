# 2026-07-09 Peg-insert dataset/interface preparation

## Status

Prepared the peg-insert pipeline up to the point where demonstrations can be collected and audited.  No formal training, batch evaluation, or dataset overwrite was started.

## Environment/interface checks

- Task: `Isaac-Peg-Insert-Franka-IK-Rel-v0`
- Control interface: relative IK + binary gripper
- Policy state: `[49]`
- Action: `[7]`
- RGB observations now exposed in the environment:
  - `rgb_camera.table_cam`: `[84,84,3]` uint8
  - `rgb_camera.wrist_cam`: `[84,84,3]` uint8
- LeRobot conversion target:
  - `observation.state`: `[49]`
  - `action`: `[7]`
  - `observation.images.camera1`: `[3,224,224]` from `rgb_table/table_cam`
  - `observation.images.camera2`: `[3,224,224]` from `rgb_wrist/wrist_cam`
  - `observation.gripper_torque`: causal `[30,1]`, newest at index `-1`

This is intentionally not compatible with the pick-place dataset contract (`state[9]`, `action[8]`). Peg-insert needs its own dataset and checkpoint configuration.

## Changes prepared

- `experiment/restore_peg_insert_assets.sh`
  - Restores Factory peg/hole USDs.
  - Restores SeattleLabTable USDs/textures.
  - Now also restores FrankaEmika Panda USD dependencies required by the peg-insert task.
- Runtime IsaacLab patches:
  - Peg/hole/table asset paths can be mapped through `LOCAL_ISAAC_4_5_ASSET_ROOT`.
  - Franka Panda asset path can be mapped through `LOCAL_ISAAC_4_5_ASSET_ROOT`.
  - `rgb_camera` observation group now includes both `table_cam` and `wrist_cam`.
- Patch archives:
  - `patches/2026-07-09_peg_insert_local_asset_root.patch`
  - `patches/2026-07-09_franka_local_asset_root.patch`
- Dataset scripts:
  - `experiment/record_peg_insert_demos.py`
    - Custom peg-insert teleop recorder.
    - Records synchronized `state`, `actions`, `rgb_table`, `rgb_wrist`, and `gripper_torque`.
    - Needed because IsaacLab generic `record_demos.py` does not record the separate `rgb_camera` observation group.
  - `experiment/collect_peg_insert_hdf5.sh`
    - Shell wrapper for the custom recorder.
    - Defaults to USD geometry and local Isaac 4.5 asset mirror.
  - `experiment/inspect_peg_insert_hdf5.py`
    - Raw HDF5 schema audit before conversion.
  - `experiment/convert_peg_insert_hdf5_to_lerobot.py`
    - Converts raw peg-insert HDF5 into LeRobot v3 with resized 224x224 two-camera images and causal torque window.

## Smoke test results

Command shape used:

```bash
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
PEG_INSERT_PROCEDURAL_ASSETS=0 \
PEG_INSERT_SIMPLE_TABLE=0 \
ENABLE_CAMERAS=1 \
NUM_STEPS=2 \
TIMEOUT_SECONDS=240 \
DUMP_AFTER_S=120 \
ACTION_MODE=zero \
bash experiment/peg_insert_env_smoke.sh
```

Result: PASS.

Key observed lines:

- Observation manager:
  - `policy` shape `(49,)`
  - `rgb_camera.table_cam` shape `(84, 84, 3)`
  - `rgb_camera.wrist_cam` shape `(84, 84, 3)`
- Runtime observation space:
  - `policy`: `(1,49)`
  - `rgb_camera.table_cam`: `(1,84,84,3)`
  - `rgb_camera.wrist_cam`: `(1,84,84,3)`
- Action space:
  - `(1,7)`
- Image tensors were finite and non-constant enough for smoke:
  - `table_cam` reset mean approximately `120.47`
  - `wrist_cam` reset mean approximately `78.91`

## LeRobot import check

Using `_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python`:

- `h5py`: import OK
- `PIL`: import OK
- `lerobot`: import OK
- `LeRobotDataset`: import OK

## Important caveats

1. Demonstrations are not yet collected.
2. The custom recorder is designed for successful teleop episodes only; it does not synthesize scripted demonstrations.
3. The peg-insert task should not reuse pick-place checkpoints directly because state/action dimensions differ.
4. Official SmolVLA pretraining can be used as initialization for a new peg-insert policy configuration, but the dataset interface must be peg-insert-specific.
5. Isaac warnings observed during smoke are expected/nonfatal for this setup:
   - Isaac Sim 4.5 thread-local stage compatibility warnings.
   - Factory hole fixed-joint warning.
   - Franka spatial tendon compatibility warning.
   - DLSS warning for small 84x84 render resolution.

## Follow-up data-chain smoke on 2026-07-09

Requested next step was to collect one successful teleop demo and build a dataset.

### Human teleop attempt

An interactive GUI collection was attempted with:

```bash
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
NUM_DEMOS=1 \
TELEOP_DEVICE=keyboard \
GEOMETRY_MODE=usd \
HEADLESS=0 \
DEVICE=cuda:0 \
OUT_DIR="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_teleop_smoke_20260709_1demo" \
bash experiment/collect_peg_insert_hdf5.sh
```

Findings:

- First GUI attempt used the default Isaac Sim 5.0 experience and failed dependency resolution.
- `experiment/collect_peg_insert_hdf5.sh` was updated to default GUI collection to `isaacsim_4_5/isaaclab.python.rendering.kit`.
- The second GUI attempt initialized Vulkan/Kit, then was killed by the OS with exit `137` before entering the teleop recorder loop.
- No teleop HDF5 was produced.

Interpretation: on this machine, full Isaac GUI + rendering is too memory-heavy/reliable for unattended collection. For real demonstrations, use a lighter workstation/session, increase swap/RAM, or collect with SpaceMouse in an interactive Isaac session after closing other GPU/desktop-heavy applications.

### Scripted smoke dataset

To continue validating the data construction path without pretending it is real teleop data, a clearly marked scripted smoke episode was generated with:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
PEG_INSERT_PROCEDURAL_ASSETS=0 \
PEG_INSERT_SIMPLE_TABLE=0 \
timeout 180 \
_runtime/remote_handoff_gripper_lstm_work/.conda/isaaclab/bin/python \
  experiment/record_peg_insert_scripted_smoke.py \
  --dataset_file "$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_scripted_smoke_20260709_1demo/peg_insert_demos.hdf5" \
  --num_steps 80 \
  --fps 20 \
  --device cuda:0 \
  --headless \
  --enable_cameras \
  --experience isaacsim_4_5/isaaclab.python.headless.rendering.kit
```

This scripted smoke directly moves the peg pose and is only for validating data plumbing. It is not a substitute for teleop demonstrations and should not be used for final training conclusions.

Raw HDF5:

- Path: `_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_scripted_smoke_20260709_1demo/peg_insert_demos.hdf5`
- Size: approximately `1.7M`
- Audit command:

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
  experiment/inspect_peg_insert_hdf5.py \
  _runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_scripted_smoke_20260709_1demo/peg_insert_demos.hdf5
```

Raw audit result:

- demos: `1`
- total steps: `80`
- state: `[49]`
- action: `[7]`
- gripper torque: `[1]`
- `rgb_table`: `(84,84,3)`
- `rgb_wrist`: `(84,84,3)`

Converted LeRobot dataset:

- Path: `_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-franka-scripted-smoke-v1`
- Repo id used locally: `Dleshers/peg-insert-franka-scripted-smoke-v1`
- Frames: `79` after `--drop-terminal-frame`
- Post-conversion schema audit:
  - `observation.state`: `[49]`
  - `action`: `[7]`
  - `observation.images.camera1`: `[3,224,224]`
  - `observation.images.camera2`: `[3,224,224]`
  - `observation.gripper_torque`: `[30,1]`

Conversion command:

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
  experiment/convert_peg_insert_hdf5_to_lerobot.py \
  --input _runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_scripted_smoke_20260709_1demo/peg_insert_demos.hdf5 \
  --output-dir _runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets \
  --repo-id Dleshers/peg-insert-franka-scripted-smoke-v1 \
  --drop-terminal-frame
```

### Scripted/oracle robot-action collection attempt

To match the pick-and-place data source more closely, a true robot-action oracle collector was added:

- Script: `experiment/record_peg_insert_oracle_demos.py`
- Collection mode: `scripted_oracle_robot_action`
- It sends actual 7D relative-IK robot actions:
  - `delta_pos(3)`
  - `delta_rot(3)`
  - `gripper_binary(1)`
- It does not teleport the peg.
- It only exports successful episodes.

Test command:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
PEG_INSERT_PROCEDURAL_ASSETS=0 \
PEG_INSERT_SIMPLE_TABLE=0 \
timeout 360 \
_runtime/remote_handoff_gripper_lstm_work/.conda/isaaclab/bin/python \
  experiment/record_peg_insert_oracle_demos.py \
  --dataset_file "$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_robot_20260709_1demo/peg_insert_demos.hdf5" \
  --num_demos 1 \
  --max_attempts 5 \
  --max_steps 260 \
  --fps 20 \
  --device cuda:0 \
  --headless \
  --enable_cameras \
  --experience isaacsim_4_5/isaaclab.python.headless.rendering.kit
```

Result: no successful robot-action demo was collected.

- Output file exists but contains no demos:
  - `_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_robot_20260709_1demo/peg_insert_demos.hdf5`
  - `data` group exists
  - `data.attrs["total"] == 0`
  - `data.keys() == []`
- The collector correctly refused to export failed attempts.

Observed failure mode:

- Peg stayed near its initial table pose and was only nudged slightly.
- The simple approach/close/lift/insert phases did not achieve a stable grasp.
- The low-dimensional hole observation in the policy vector appeared as approximately `[0,0,0]` during attempts, while peg positions were around `[0.50,-0.08,*]`, `[0.55,-0.10,*]`, or `[0.60,-0.12,*]`. This needs a dedicated peg/hole observation audit before using it for closed-loop oracle targeting.

Conclusion:

- The data plumbing is ready.
- Scripted smoke can validate schema, cameras, torque windows, and LeRobot conversion.
- A true pick-and-place-style robot-action scripted expert for peg-insert is not yet solved. It needs further work on:
  1. peg/hole observation semantics;
  2. end-effector pose offsets for grasping the tiny peg;
  3. gripper close timing and contact stability;
  4. success predicate consistency for Factory peg/hole assets.

Post-conversion loading required writable Hugging Face caches:

```bash
HF_HOME="$PWD/_runtime/remote_handoff_gripper_lstm_work/.cache/huggingface" \
HF_DATASETS_CACHE="$PWD/_runtime/remote_handoff_gripper_lstm_work/.cache/huggingface/datasets" \
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python <audit-snippet>
```

## Recommended next command

For a real human teleop dataset smoke, collect 1 successful demo in an interactive session with enough memory/swap:

```bash
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
NUM_DEMOS=1 \
TELEOP_DEVICE=keyboard \
GEOMETRY_MODE=usd \
HEADLESS=0 \
bash experiment/collect_peg_insert_hdf5.sh
```

Then audit it before conversion:

```bash
python3 experiment/inspect_peg_insert_hdf5.py <path-to-peg_insert_demos.hdf5>
```

If the raw audit passes, convert with:

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
  experiment/convert_peg_insert_hdf5_to_lerobot.py \
  --input <path-to-peg_insert_demos.hdf5> \
  --output-dir _runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets \
  --repo-id Dleshers/peg-insert-franka-visual-torque-smoke-v1 \
  --drop-terminal-frame
```
