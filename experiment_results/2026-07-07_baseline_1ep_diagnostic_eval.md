# 2026-07-07 baseline 1-episode 诊断 eval

## 运行目的

在本地 Isaac eval server 已监听 `tcp://*:5562` 后，运行 1 个 pure SmolVLA baseline 诊断 episode，保存视频、`eval_info.json` 和 server trajectory JSONL，用于分析 baseline 闭环 0% success 的失败模式。

本次不是批量评估；未重新训练、未覆盖数据集或 checkpoint。

## 运行配置

- policy：`_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- server：`localhost:5562`
- env type：`isaaclab_tactile_remote`
- control mode：`joint`
- episodes：`1`
- seed：`1000`
- policy device：`cuda`
- `chunk_size`：`50`
- `n_action_steps`：`50`
- `load_vlm_weights`：`false`

说明：

- 第一次 sandbox 内尝试时 CUDA 不可见，且 policy 尝试访问 Hugging Face public VLM config/weights；该尝试已中断。
- 正式诊断 run 使用非 sandbox 执行，CUDA 可用，并设置 `--policy.load_vlm_weights=false`，从本地 baseline checkpoint 加载权重。

## 输出文件

诊断目录：

`_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_seed1000_20260707_0629/`

关键文件：

- `eval_info.json`
- `baseline_probe_trajectory.jsonl`
- `trajectory_summary.json`
- `videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`
- `frames/frame_001.png` ... `frames/frame_006.png`

注意：server 的原始相对 `TRAJECTORY_LOG` 被写到了 IsaacLab-Tactile 工作目录下的嵌套路径，已复制到诊断目录：

`baseline_probe_trajectory.jsonl`

## eval 结果

`eval_info.json`：

- `n_episodes`: `1`
- `pc_success`: `0.0`
- `avg_sum_reward`: `0.0`
- `avg_max_reward`: `0.0`
- episode runtime：约 `43.6s`
- video：`eval_episode_0.mp4`

## trajectory 摘要

trajectory 行数：

- `301` 行：`reset` + `300` step

任务结果：

- success：`false`
- done step：`300`
- reward sum：`0.0`

cube：

- initial cube pos：`[0.3861058, -0.0917004, 0.02199999]`
- final cube pos：`[0.3836191, -0.0940755, 0.02199996]`
- final cube displacement：`0.00344 m`
- max cube displacement：`0.02340 m`
- max cube lift delta：`0.01098 m`
- final cube XY distance to basket：`0.26882 m`

gripper：

- first negative gripper command step：`61`
- first gripper qpos below `0.005` step：`66`
- final gripper qpos：`[5.8e-7, 0.0]`
- gripper command range：`[-1.0541, 1.0440]`

joint tracking:

- joint target error mean：`0.07136`
- p90：`0.11952`
- max：`0.64156`

action range:

- arm action dimensions remain within normal training distribution ranges.
- gripper slightly overshoots training range `[-1, 1]`, but IsaacLab binary gripper uses sign, so this is not enough to explain failure.

## 50-step chunk behavior observed

The run reflects the expected SmolVLA queue behavior: one inference produces a 50-step action chunk.

Per 50-step block:

| Steps | Gripper cmd mean | Cube max displacement | Cube max z | Joint err mean |
| --- | ---: | ---: | ---: | ---: |
| 1-50 | `+1.014` | `~0.00000 m` | `0.02200 m` | `0.0553` |
| 51-100 | `-0.613` | `0.02340 m` | `0.03298 m` | `0.0821` |
| 101-150 | `-0.982` | `0.00344 m` | `0.02200 m` | `0.0867` |
| 151-200 | `-0.879` | `0.00344 m` | `0.02200 m` | `0.0651` |
| 201-250 | `-0.0218` | `0.00344 m` | `0.02200 m` | `0.0899` |
| 251-300 | `+0.358` | `0.00344 m` | `0.02200 m` | `0.0490` |

Interpretation:

- The policy keeps the gripper open for the first 50 steps.
- It begins closing at step 61.
- The gripper physically closes by step 66.
- The cube is nudged/lifted only slightly around steps 51-100, then returns/remains near the tabletop.
- The cube is never transported toward the basket.

## 视频观察

抽帧显示：

- initial cube / basket placement appears normal.
- table-camera video is nearly static from the cube's perspective.
- cube remains near its start location and is not visibly carried to the basket.

Limitation:

- The saved video is table camera only and does not clearly show the gripper/end-effector relative to cube.
- Current trajectory JSONL lacks eef pose, so this run cannot precisely determine whether the gripper was above, beside, or behind the cube at close time.

## 当前失败模式判断

This episode is most consistent with:

1. the robot did not bring the gripper to a successful grasp pose;
2. the gripper did close, but not with the cube stably captured;
3. the cube was nudged/slightly lifted at most, then remained on or near the tabletop;
4. there is no evidence of action-scale explosion or gripper sign inversion.

This strengthens the earlier hypothesis:

- baseline checkpoint and processors are valid;
- closed-loop failure is likely from visual/action closed-loop drift and/or long `n_action_steps=50` open-loop chunks;
- more precise diagnosis needs eef pose in trajectory telemetry.

## Recommended next step

Patch `eval_server.py` trajectory logging to include:

- `eef_pos_before`
- `eef_pos_after`
- `eef_cube_dist_before`
- `eef_cube_dist_after`
- optionally gripper command sign and action chunk index

Then rerun a single 1-episode baseline diagnostic with the same baseline config. This will directly answer whether failure is approach error, close-at-wrong-location, push-away, or lift/drop.

