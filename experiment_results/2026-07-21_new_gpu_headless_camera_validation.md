# New GPU headless camera validation (2026-07-21)

## Outcome

The GPU/storage migration fixed the Isaac Sim headless RGB camera blocker. On the new machine, true RGB camera capture works when using the restored official Replicator/viewport dependency chain plus a custom rendering kit that disables NGX/DLSS.

## Machine

```text
GPU: NVIDIA GeForce RTX 4090
Driver: 560.35.03
CUDA reported by nvidia-smi: 12.6
Disk: /root/autodl-tmp 100G, ~55G free during validation
```

## Runtime adjustments

Restored old diagnostic edits before testing:

- restored `omni.replicator.core` dependency on `omni.kit.viewport.window`;
- restored `omni.kit.widget.viewport/.../texture.py` to remove the old `PEG_INSERT_SKIP_VIEWPORT_WIDGET_HYDRA` diagnostic bypass.

Created runtime kit:

```text
_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit
```

This is copied from `isaaclab.python.headless.rendering.kit` and appends:

```toml
[settings]
ngx.enabled = false
rtx.post.dlss.enabled = false
rtx-transient.dlssg.enabled = false
rtx.post.dlss.execMode = 0
```

Without this NGX/DLSS disablement, the official rendering kit did not segfault on the new driver but hung after `Failed to create NGX context`.

## Tests passed

### 1. Standalone Replicator RGB probe

Command used `experiment/replicator_min_camera_probe.py` with:

```text
--headless --enable_cameras --experience=isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit
```

Log:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/logs/replicator_min_camera_new_gpu_nongx.log
```

Result: 10 consecutive non-empty RGB(A) frames:

```text
shape=(84, 84, 4), dtype=uint8, size=28224, min=0, max=255
```

### 2. Peg-insert camera smoke

Updated `experiment/peg_insert_headless_probe.py` to dynamically expose both `wrist_cam` and `table_cam` in `rgb_camera` when cameras are enabled.

Log:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/logs/peg_insert_env_smoke_new_gpu_camera_nongx_tablecam.log
```

Result: both task cameras are non-empty:

```text
reset_obs.rgb_camera.wrist_cam: shape=(1, 84, 84, 3), uint8, min=0, max=254
reset_obs.rgb_camera.table_cam: shape=(1, 84, 84, 3), uint8, min=0, max=249
obs_step_3.rgb_camera.wrist_cam: shape=(1, 84, 84, 3), uint8
obs_step_3.rgb_camera.table_cam: shape=(1, 84, 84, 3), uint8
```

### 3. Oracle RGB HDF5 smoke collection

Collected one `preinsert_alignment` demo with true RGB cameras enabled.

Raw HDF5:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_rgb_smoke_20260721/peg_insert_demos.hdf5
```

Result:

```text
demo_0: 178 steps
file size: 2.5M
rgb_table: (178, 84, 84, 3), uint8, min=0, max=255, mean=135.92
rgb_wrist: (178, 84, 84, 3), uint8, min=0, max=255, mean=110.22
state: (178, 49), float32
actions: (178, 7), float32
gripper_torque: (178, 1), float32
```

`experiment/inspect_peg_insert_hdf5.py` passed:

```text
OK: demos=1 total_steps=178
OK: state=[49] action=[7] gripper_torque=[1]
OK: rgb_table image_shapes=[(84, 84, 3)]
OK: rgb_wrist image_shapes=[(84, 84, 3)]
```

### 4. LeRobot conversion/audit

Converted the RGB HDF5 smoke dataset to local LeRobot format:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-oracle-rgb-smoke-20260721-compact21-v1
```

Conversion result:

```text
frames=178
Interface: state=[21], action=[7], camera1/2=[3,224,224], torque=[30,1]
size=5.2M
```

Feature audit passed:

```text
OK observation.state shape=(21,) dtype=torch.float32
OK action shape=(7,) dtype=torch.float32
OK observation.images.camera1 shape=(3, 224, 224) dtype=torch.float32
OK observation.images.camera2 shape=(3, 224, 224) dtype=torch.float32
OK observation.gripper_torque shape=(30, 1) dtype=torch.float32
```

## Current recommended command pattern

Use this experience for future camera-enabled peg-insert work on this machine:

```bash
EXPERIENCE=isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit
```

Example smoke:

```bash
ROOT=/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla/_runtime/remote_handoff_gripper_lstm_work \
RUNTIME_ROOT=$ROOT ENABLE_CAMERAS=1 NUM_STEPS=4 TIMEOUT_SECONDS=360 DUMP_AFTER_S=120 \
EXPERIENCE=isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit \
bash experiment/peg_insert_env_smoke.sh
```

## Next work

1. Collect a larger RGB peg-insert oracle dataset with this `nongx` kit.
2. Convert to LeRobot compact21 with torque windows.
3. Run a short training smoke on the RGB dataset.
4. Then scale data collection/training/evaluation for insert-focused visual vs torque-LSTM comparison.
