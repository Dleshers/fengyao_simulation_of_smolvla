# 2026-07-09 Peg-insert oracle dataset build notes

## Objective

Prepare a scripted/oracle peg-insert dataset path analogous to the earlier
pick-and-place scripted data collection, with the following constraints:

- headless Isaac Lab collection,
- robot-action labels, not human teleop,
- RGB observations enabled for the final raw HDF5,
- LeRobot conversion with the project tactile window interface.

## Key compatibility findings

1. **Factory USD hole is not currently safe as a `RigidObject` target.**
   A no-camera pose audit with `PEG_INSERT_PROCEDURAL_ASSETS=0` showed that the
   hole root pose is correct at reset step 0, but after one simulation step it
   jumps from approximately `(0.55, 0.0, 0.0)` to near `(0, 0, 0)`. Isaac also
   reports:

   - `PhysicsUSD: CreateJoint - found a joint with disjointed body transforms`
     for `/World/envs/env_0/Hole/forge_hole_8mm/FixedJoint`.

   This means the current Factory hole USD cannot be trusted for observation
   `hole_pos`, oracle target generation, or success checking without a separate
   static-asset/pose-reference fix.

2. **Procedural peg/hole root semantics are consistent.**
   A no-camera pose audit with `PEG_INSERT_PROCEDURAL_ASSETS=1` and
   `PEG_INSERT_SIMPLE_TABLE=1` confirmed that:

   - policy `peg_pos`, `hole_pos`, and `peg_to_hole_pos` match Isaac Lab asset
     root poses;
   - state shape remains `[49]`;
   - action space remains `[7]`.

3. **Strict insertion success is not physically meaningful for the procedural
   target block.**
   The procedural "hole" is currently a solid cuboid stand-in, not a collision
   mesh with an actual opening. Forcing the peg below the target surface causes
   the peg to slip/push away. Therefore a separate collection success mode was
   added:

   - `preinsert_alignment`: peg is grasped, moved over the target, and lowered
     close to the target top without forcing it through the solid cuboid.

   This dataset is suitable for validating robot-action oracle collection,
   visual/state/tactile interface wiring, and conversion. It should not be
   presented as a completed physical insertion benchmark.

## Code changes used

- Added `experiment/audit_peg_insert_scene_pose.py`.
- Updated `experiment/record_peg_insert_oracle_demos.py`:
  - supports `--allow_missing_images` for no-camera action debugging only;
  - logs eef/target/peg/hole diagnostics;
  - anchors peg/hole targets at episode reset instead of chasing the live peg
    pose during lift/transport;
  - adds `--success_mode {inserted,preinsert_alignment}`;
  - exits with `os._exit(2)` on collection failure to avoid Isaac cleanup hangs.

## Successful raw RGB HDF5

Command class:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
PEG_INSERT_DISABLE_CAMERAS=0 \
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
PEG_INSERT_PROCEDURAL_ASSETS=1 \
PEG_INSERT_SIMPLE_TABLE=1 \
timeout 220 \
_runtime/remote_handoff_gripper_lstm_work/.conda/isaaclab/bin/python \
  experiment/record_peg_insert_oracle_demos.py \
  --dataset_file "$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_preinsert_rgb_20260709_1demo/peg_insert_demos.hdf5" \
  --num_demos 1 \
  --max_attempts 4 \
  --max_steps 235 \
  --fps 20 \
  --device cuda:0 \
  --headless \
  --enable_cameras \
  --success_mode preinsert_alignment \
  --experience isaacsim_4_5/isaaclab.python.headless.rendering.kit
```

Result:

- Saved `demo_0` on attempt 2.
- Raw path:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_preinsert_rgb_20260709_1demo/peg_insert_demos.hdf5`
- Frames: 182 raw steps.
- Collection mode: `scripted_oracle_robot_action`.
- Success mode: `preinsert_alignment`.

Raw HDF5 audit:

- demos: 1
- total steps: 182
- state: `[49]`
- action: `[7]`
- `rgb_table`: `(84,84,3)`
- `rgb_wrist`: `(84,84,3)`
- `gripper_torque`: `[1]`
- torque range: approximately `[-80, 3.32]`

## LeRobot conversion

Command:

```bash
HF_HOME="$PWD/_runtime/remote_handoff_gripper_lstm_work/.cache/huggingface" \
HF_DATASETS_CACHE="$PWD/_runtime/remote_handoff_gripper_lstm_work/.cache/huggingface/datasets" \
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
  experiment/convert_peg_insert_hdf5_to_lerobot.py \
  --input _runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_preinsert_rgb_20260709_1demo/peg_insert_demos.hdf5 \
  --output-dir _runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets \
  --repo-id Dleshers/peg-insert-franka-oracle-preinsert-rgb-v1 \
  --drop-terminal-frame
```

Result:

- Local dataset:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/Dleshers/peg-insert-franka-oracle-preinsert-rgb-v1`
- Episodes: 1
- Frames: 181
- `observation.state`: `[49]`, `float32`
- `action`: `[7]`, `float32`
- `observation.images.camera1`: `[3,224,224]`, `float32`
- `observation.images.camera2`: `[3,224,224]`, `float32`
- `observation.gripper_torque`: `[30,1]`, `float32`

## Remaining blockers before a formal peg-insert benchmark dataset

1. Replace or repair the Factory hole asset usage so the physical hole remains
   static at the configured target pose and the policy observation uses the same
   target pose.
2. Replace the procedural solid target block with an actual insertable collision
   geometry if using procedural assets for the benchmark.
3. Define a physically meaningful success predicate for root semantics:
   root-center vs peg tip/base must be explicit.
4. Increase swap before larger RGB collection. During tests, swap free space
   dropped below 200 MB, which makes Isaac/Kit more fragile.

## Recommended next command

For a small interface-only dataset expansion after confirming the above caveat:

```bash
OMNI_KIT_ACCEPT_EULA=YES PEG_INSERT_DISABLE_CAMERAS=0 \
LOCAL_ISAAC_4_5_ASSET_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/assets/isaac_4_5_mirror" \
PEG_INSERT_PROCEDURAL_ASSETS=1 PEG_INSERT_SIMPLE_TABLE=1 \
timeout 900 \
_runtime/remote_handoff_gripper_lstm_work/.conda/isaaclab/bin/python \
  experiment/record_peg_insert_oracle_demos.py \
  --dataset_file "$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/raw_hdf5/peg_insert_oracle_preinsert_rgb_20260709_5demo/peg_insert_demos.hdf5" \
  --num_demos 5 --max_attempts 20 --max_steps 235 --fps 20 \
  --device cuda:0 --headless --enable_cameras \
  --success_mode preinsert_alignment \
  --experience isaacsim_4_5/isaaclab.python.headless.rendering.kit
```
