# 2026-07-08 Targeted Collection Feasibility Smoke

## Scope

This smoke test checked whether the existing scripted TacEx data-collection pipeline can collect a small successful expert batch at a fixed cube pose, for possible targeted dataset augmentation after baseline closed-loop failures.

No existing dataset, checkpoint, or Hugging Face asset was overwritten. The generated HDF5 is intentionally kept under ignored `_runtime/` storage.

## Git state before collection

- Pushed commit before this smoke: `88e079b` (`Document baseline eval diagnostics and success plumbing fixes`).
- Added a diagnostic-only `--fixed_cube_pose` hook in the runtime state-machine collector.
- Added optional `FIXED_CUBE_POSE` passthrough to `experiment/rebuild_dataset.sh`.
- Archived the intended code delta in `patches/2026-07-08_targeted_collection_fixed_cube_pose.patch`.

## Command

```bash
PERSISTENT_ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent" \
RUN_ID="targeted_fixedcube_0425_m0115_n2_seed2400_20260708" \
NUM_DEMOS=2 \
NUM_ENVS=1 \
SEED=2400 \
RUN_ISAAC_SMOKE=0 \
FIXED_CUBE_POSE="0.425,-0.115,0.022" \
bash _runtime/remote_handoff_gripper_lstm_work/experiment/run_hdf5_smoke.sh
```

## Output

- Raw HDF5:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted_fixedcube_0425_m0115_n2_seed2400_20260708/data.hdf5`
- Collection log:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted_fixedcube_0425_m0115_n2_seed2400_20260708/collection.log`
- SHA256:
  `e09ed3a02fedc4a96af6b8d141b3ea0d34373438e954ca05ea02415bf1b23c77`

## Result

- `inspect_raw_hdf5.py`: `OK: demos=2 total_steps=416`
- Root metadata:
  - `total_demos=2`
  - `attempted_episodes=2`
  - `failed_or_filtered_episodes=0`
  - `state_schema=joint_state_9d:[arm_joint_pos(7),gripper_qpos(2)]`
  - `action_schema=joint_action_8d:[arm_joint_pos_target_abs(7),gripper_cmd(1)]`
  - `fps=20`
- Both episodes have `success=true`.
- Both episodes start at exactly the requested env-frame cube pose:
  `[0.4250000119, -0.1150000021, 0.0219999999]`
- Demo lengths:
  - `demo_0`: 206 steps
  - `demo_1`: 210 steps
- RGB observations are present:
  - `rgb_table`: `[steps,224,224,3]`
  - `rgb_wrist`: `[steps,224,224,3]`
- Actions are present as 8D joint targets plus scalar gripper command:
  - `demo_0`: `[206,8]`
  - `demo_1`: `[210,8]`
  - gripper command values observed: `[-1.0, 1.0]`
- Gripper torque is present:
  - shape `[steps,1]`
  - observed range about `[-45.28, 3.57]`
- The cube was lifted and delivered near the basket:
  - `demo_0`: `cube_z_max=0.2118`, final cube-basket XY distance about `0.0186 m`
  - `demo_1`: `cube_z_max=0.2182`, final cube-basket XY distance about `0.0087 m`

## Notes

- Isaac still aborts during camera/tiled-camera shutdown with the known weak-reference cleanup error:
  `ReferenceError: weakly-referenced object no longer exists`
- The wrapper treated this as non-fatal because a finalized non-empty HDF5 existed and passed raw inspection.
- During scene loading, Isaac logged missing remote material/texture warnings from NVIDIA Omniverse content. These did not block collection.

## Conclusion

Small targeted data collection is feasible. The expert state machine can generate successful demonstrations at a fixed pose representative of the policy's problematic region. This supports the next step: run a small, isolated targeted augmentation batch over several failure poses, convert it to a separate LeRobot dataset, and evaluate whether mixing it with the original dataset improves closed-loop baseline behavior.

## Suggested next command

Run a slightly broader isolated batch over a small list of baseline-failure poses, still without merging into the official dataset:

```bash
# Repeat the smoke command with 3-5 selected fixed poses and unique RUN_ID/RAW_DIR values,
# then convert each output to separate LeRobot repos for audit before any training merge.
```
