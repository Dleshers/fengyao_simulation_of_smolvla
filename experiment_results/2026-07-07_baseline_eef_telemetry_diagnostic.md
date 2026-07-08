# 2026-07-07 baseline eef telemetry 诊断 eval

## 目的

在上一轮 1-episode baseline 诊断中，trajectory 已证明 gripper 会闭合，但缺少 end-effector 位姿，无法判断 close 发生时夹爪是否真正接近 cube。

本轮对 runtime `IsaacLab-Tactile/scripts/eval_server.py` 增加只读 telemetry 字段后，重新运行 1 个 baseline episode：

- `eef_pos_before`
- `eef_pos_after`
- `eef_cube_dist_before`
- `eef_cube_dist_after`
- `eef_basket_dist_before`
- `eef_basket_dist_after`
- `action_gripper_cmd`
- `action_gripper_sign`
- `action_chunk_index_50`

本次仍不是批量评估；未训练、未改 checkpoint、未覆盖数据。

## 运行配置

- server port：`5563`
- control mode：`joint`
- policy：`_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- episodes：`1`
- seed：`1000`
- policy device：`cuda`
- `--policy.load_vlm_weights=false`
- `chunk_size=50`
- `n_action_steps=50`

输出目录：

`_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1000_20260707_0643/`

关键文件：

- `eval_info.json`
- `baseline_eef_trajectory.jsonl`
- `eef_trajectory_summary.json`
- `videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`

## 运行结果

`eval_info.json`：

- episodes：`1`
- success：`false`
- `pc_success`: `0.0`
- `avg_sum_reward`: `0.0`
- runtime：约 `40.5s`

server cleanup 时出现 Isaac/Replicator 析构异常：

- `ReferenceError: weakly-referenced object no longer exists`

这发生在 episode 完成和 telemetry 写出之后；GPU 已释放，不影响本次数据有效性。

## eef / cube 关键量化结果

trajectory：

- rows：`301`，即 `reset + 300 steps`
- done step：`300`
- success：`false`

cube：

- initial cube pos：`[0.3861058, -0.0917004, 0.02199999]`
- final cube pos：`[0.3832670, -0.0944681, 0.02199986]`
- final cube displacement：`0.00396 m`
- max cube displacement：`0.02369 m`
- max cube lift delta：`0.01132 m`

gripper：

- first negative gripper command step：`61`
- first physical closed qpos step：`66`
- gripper command range：`[-1.0534, 1.0423]`

eef-cube distance:

- initial eef-cube distance：`0.25557 m`
- minimum eef-cube distance：`0.07436 m` at step `63`
- eef-cube distance at first negative gripper command：`0.07723 m`
- eef-cube distance when gripper qpos first below `0.005`：`0.07568 m`
- final eef-cube distance：`0.09411 m`

## 分阶段观察

| Step window | eef-cube min | eef-cube mean | cube max disp | cube max lift | gripper mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-50 | `0.12187 m` | `0.19050 m` | `~0 m` | `~0 m` | `+1.013` |
| 51-100 | `0.07436 m` | `0.08322 m` | `0.02369 m` | `0.01132 m` | `-0.614` |
| 101-150 | `0.10406 m` | `0.19892 m` | `0.00396 m` | `~0 m` | `-0.983` |
| 151-300 | `0.09397 m` | `0.20576 m` | `0.00396 m` | `~0 m` | `-0.254` |

Around close:

- steps 55-75:
  - eef-cube min：`0.07436 m`
  - eef-cube mean：`0.07918 m`
  - cube max displacement：`0.00843 m`
  - cube max lift：`0.00624 m`
  - gripper qpos at step 75：`~0.00268`

## 失败模式判断

本轮 eef telemetry 直接回答了上一轮缺口：

- gripper 不是没闭合；
- action scale 没有明显爆炸；
- cube 不是被抓起后掉落；
- cube 不是被送到 basket 附近后 release 失败；
- close 发生时 eef-cube 距离仍约 `7.5 cm`，且 cube 只被轻微推动/抬起。

因此，本 episode 最符合：

1. approach / grasp pose 没到位；
2. close command 发生在未形成稳定夹取几何的位置；
3. long `n_action_steps=50` chunk 可能使 close/replan 时机过粗，导致错过精细接触修正；
4. baseline 离线拟合可用，但闭环接触阶段发生 covariate shift。

## 建议下一步

做一个仍然只有 1 episode 的对照，不作为正式指标：

- 保持同一 checkpoint、同一 seed、同一 server telemetry；
- 临时将 eval-time `n_action_steps` override 为 `1` 或小值；
- 比较：
  - min eef-cube distance；
  - eef-cube distance at first close；
  - cube lift/displacement；
  - success 或是否形成稳定 grasp。

如果短 horizon 明显改善 eef-cube 接近或夹取时机，就能把主要问题锁定到长 open-loop chunk；如果仍然停在约 `7-8 cm`，则更可能是视觉 policy 本身的闭环 grasp pose 预测偏差。

