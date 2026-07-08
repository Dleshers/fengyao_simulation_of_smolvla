# 2026-07-08 Targeted LeRobot Conversion and Independent Audit

## Scope

Convert the small fixed-pose targeted raw HDF5 smoke dataset into a separate LeRobot dataset and audit it independently. This was intentionally isolated from the official 200-episode dataset and from any training workflow.

## Inputs

- Raw HDF5:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted_fixedcube_0425_m0115_n2_seed2400_20260708/data.hdf5`
- Raw HDF5 SHA256:
  `e09ed3a02fedc4a96af6b8d141b3ea0d34373438e954ca05ea02415bf1b23c77`
- Fixed cube pose:
  `[0.425, -0.115, 0.022]`
- Raw demos:
  `2`
- Raw frames:
  `416`

## First conversion

Command:

```bash
RAW_DIR="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted_fixedcube_0425_m0115_n2_seed2400_20260708" \
DATASET_PARENT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/datasets" \
DATASET_REPO_ID="targeted_fixedcube_0425_m0115_w30_n2_seed2400_v1" \
_runtime/remote_handoff_gripper_lstm_work/experiment/rebuild_dataset.sh convert
```

Output:

- `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/targeted_fixedcube_0425_m0115_w30_n2_seed2400_v1`

Standard validation passed:

- `total_episodes=2`
- `total_frames=416`
- `observation.state`: `[9]`
- `observation.images.camera1`: `[3,224,224]`
- `observation.images.camera2`: `[3,224,224]`
- `action`: `[8]`
- `observation.gripper_torque`: `[30,1]`
- causal torque audit: `416` sequential windows, `2` episode starts, `414` within-episode transitions

## Semantic audit finding

The first converted dataset passed format validation but failed an action-label semantic boundary check.

For each raw demo, the maximum arm action-vs-state L2 jump occurred at the final frame:

| demo | frames | final-frame action-state L2 | previous-frame L2 | final phase |
|---|---:|---:|---:|---:|
| `demo_0` | 206 | `0.6747` | `0.0592` | `13` |
| `demo_1` | 210 | `0.8851` | `0.0486` | `13` |

The final-frame action resembled the next/reset joint target rather than the release-step target. This is consistent with Isaac auto-reset updating `robot.data.joint_pos_target` before the recorder reads the post-step target at terminal success.

Because this targeted set is tiny, keeping one mislabeled terminal frame per episode would be a non-negligible contamination. The raw data was not modified.

## Clean conversion

A runtime converter patch was added and archived as:

- `patches/2026-07-08_converter_drop_terminal_frame.patch`

It adds:

```bash
--drop-terminal-frame
```

This drops the final raw frame from each demo during conversion only.

Command:

```bash
OUT_PARENT="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/datasets"
RAW="$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/smoke_joint_torque_w30_targeted_fixedcube_0425_m0115_n2_seed2400_20260708/data.hdf5"
REPO="targeted_fixedcube_0425_m0115_w30_n2_seed2400_clean_v1"

HF_HOME="$PWD/_runtime/remote_handoff_gripper_lstm_work/.cache/huggingface" \
XDG_CACHE_HOME="$PWD/_runtime/remote_handoff_gripper_lstm_work/.cache" \
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/lerobot/convert_pick_place_basket_joint_tacex.py \
  --input "$RAW" \
  --output-dir "$OUT_PARENT" \
  --repo-id "$REPO" \
  --fps 20 \
  --torque-window-size 30 \
  --use-videos \
  --drop-terminal-frame
```

Output:

- `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/targeted_fixedcube_0425_m0115_w30_n2_seed2400_clean_v1`

## Clean dataset validation

Standard validation passed:

- `total_episodes=2`
- `total_frames=414`
- causal torque audit: `414` sequential windows, `2` episode starts, `412` within-episode transitions
- `observation.state`: `[9]`
- `observation.images.camera1`: `[3,224,224]`
- `observation.images.camera2`: `[3,224,224]`
- `action`: `[8]`
- `observation.gripper_torque`: `[30,1]`
- state range: `[-2.6371, 2.91986]`
- action range: `[-2.63675, 2.92289]`
- torque range: `[-45.2751, 0.00412471]`

Clean semantic audit:

| episode | frames | final frame | mean action-state L2 | p95 L2 | max L2 | final-frame L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 205 | 204 | `0.0705` | `0.1434` | `0.1824` | `0.0592` |
| 1 | 209 | 208 | `0.0651` | `0.1274` | `0.1384` | `0.0486` |

The reset-sized terminal action jump is gone in the clean conversion.

## File layout

Clean dataset files:

- `data/chunk-000/file-000.parquet`: about `51.6 MB`
- `meta/info.json`: `2301 bytes`
- `meta/stats.json`: `15171 bytes`
- `meta/tasks.parquet`: `2291 bytes`
- `meta/episodes/chunk-000/file-000.parquet`: `56957 bytes`

The converter was invoked with `--use-videos`, but this local LeRobot version stored image features inside parquet for this tiny dataset; no separate `videos/` files were produced.

## Conclusion

An independent targeted LeRobot dataset now exists and passes both schema-level and semantic boundary audits:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/targeted_fixedcube_0425_m0115_w30_n2_seed2400_clean_v1
```

For future targeted augmentation, use the cleaned conversion path or fix the collector to snapshot the action target before terminal auto-reset. Until then, avoid training directly on the unclean `targeted_fixedcube_0425_m0115_w30_n2_seed2400_v1` conversion.

## Suggested next command

Collect 3-5 additional fixed poses as separate raw HDF5 runs, then convert each with `--drop-terminal-frame` and run the same audit before any merge or retraining.
