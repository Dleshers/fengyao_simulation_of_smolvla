# 2026-07-20 peg-insert Stage 0/1 no-camera feasibility

## Context

After the torque-disambiguation 2000-step diagnostic completed, the next experiment moved toward the contact-rich peg-insert Isaac task.  The immediate goal was to validate the simulator task, raw HDF5 collection path, and LeRobot conversion path on the expanded AutoDL disk.

Disk after expansion:

```text
/root/autodl-tmp: 100G total, about 55G available at start of this stage
```

## Stage 0: Isaac peg-insert smoke

Initial camera-enabled smoke using `isaaclab.python.headless.rendering.kit` segfaulted inside the viewport/Hydra stack:

```text
omni.kit.widget.viewport ... __enable_hydra_engine
```

The robust smoke path is now no-camera + non-rendering headless kit:

```bash
RUNTIME_ROOT=_runtime/remote_handoff_gripper_lstm_work TIMEOUT_SECONDS=240 NUM_STEPS=4 bash experiment/peg_insert_env_smoke.sh
```

Result:

```text
[PEG_PROBE] success
observation_space=Dict('policy': Box(-inf, inf, (1, 49), float32))
action_space=Box(-inf, inf, (1, 7), float32)
```

The script was updated so default smoke uses:

```text
ENABLE_CAMERAS=0
EXPERIENCE=isaacsim_4_5/isaaclab.python.headless.kit
```

When cameras are disabled, the probe explicitly removes `scene.wrist_cam`, `scene.table_cam`, and `observations.rgb_camera` from the parsed env config.

## Stage 1: no-camera scripted oracle collection

A small unattended oracle collection was run with no cameras and zero-image placeholders:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_preinsert_nocam_20260720_stage1/peg_insert_demos.hdf5
```

Command characteristics:

```text
num_demos=3
max_attempts=8
success_mode=preinsert_alignment
allow_missing_images=true
experience=isaacsim_4_5/isaaclab.python.headless.kit
```

Outcome:

```text
saved demos: 3 / 8 attempts
total raw frames: 617
```

Raw HDF5 audit:

```text
OK: demos=3 total_steps=617
OK: state=[49] action=[7] gripper_torque=[1]
OK: torque range=[-80, 0.0731096]
OK: rgb_table image_shapes=[(84, 84, 3)]
OK: rgb_wrist image_shapes=[(84, 84, 3)]
```

The RGB arrays are placeholders: both cameras have one unique value only, so this dataset must not be treated as visual training data.

## Stage 2: LeRobot conversion smoke

The raw HDF5 was converted to compact21 LeRobot format:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-oracle-preinsert-nocam-stage1-compact21-v1
```

Conversion settings:

```text
state_mode=compact21
torque_window_size=30
torque_control=original
drop_terminal_frame=true
```

LeRobot audit:

```text
episodes=3 frames=614
OK observation.state shape=(21,) dtype=torch.float32
OK action shape=(7,) dtype=torch.float32
OK observation.images.camera1 shape=(3, 224, 224) dtype=torch.float32
OK observation.images.camera2 shape=(3, 224, 224) dtype=torch.float32
OK observation.gripper_torque shape=(30, 1) dtype=torch.float32
torque_newest min=-80 max=0.0731096 mean=-5.97227 std=7.29358
torque_window min=-80 max=0.0731096 mean=-11.07 std=20.3753
```

## Interpretation

This stage validates the non-visual physical pipeline:

- Isaac peg-insert IK-Rel task can launch, reset, and step on the remote AutoDL server.
- The scripted oracle can create successful preinsert-alignment episodes with real robot IK actions.
- Raw HDF5 schema and LeRobot conversion are healthy for state/action/torque.
- The torque channel has meaningful variation and is not merely constant zero.

The remaining blocker for visual SmolVLA peg-insert training/evaluation is headless camera rendering.  The current server fails with rendering kit viewport/Hydra segfault, while non-rendering kit lacks the viewport module needed by camera sensors.

## Recommended next step

Resolve camera capture in headless Isaac before collecting the real visual peg-insert dataset.  Practical options:

1. Create a custom headless camera experience that includes camera/viewport dependencies without triggering the viewport widget segfault.
2. Use a display-backed session if AutoDL supports a virtual X/EGL path for Isaac rendering.
3. As a fallback, continue non-visual state/torque policy experiments, but label them explicitly as non-visual controls.
