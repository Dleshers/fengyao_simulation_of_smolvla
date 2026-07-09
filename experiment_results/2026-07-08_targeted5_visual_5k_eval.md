# 2026-07-08 Targeted5 visual 5k checkpoint 闭环评估

## 结论

已完成 targeted5 visual-only 5k checkpoint 的 10 episode IsaacLab 闭环评估。

结果：

- `pc_success`: 90.0%
- `n_episodes`: 10
- success list: `[true, true, true, true, true, true, false, true, true, true]`
- 唯一失败：episode 6 / seed 1006
- `eval_s`: 317.54 s
- `eval_ep_s`: 31.75 s

这和先前 pure baseline formal dynamic RGB n=10 seed1000 的 0% success 相比，是一个非常强的正向信号：小批 targeted 数据微调显著修正了原 baseline 的闭环失败模式。

## 使用的 policy

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_ablation_visual_5000_seed1000/checkpoints/005000/pretrained_model
```

配置摘要：

- SmolVLA visual-only
- `use_tactile=false`
- `use_torque_lstm=false`
- `control_mode=joint`
- action dim: 8
- state dim: 9
- camera rename map:

```json
{
  "observation.images.rgb_table": "observation.images.camera1",
  "observation.images.rgb_wrist": "observation.images.camera2"
}
```

## 输出路径

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_ablation_visual_5k_eval_n10_seed1000_20260708
```

包含：

- `eval_info.json`
- `trajectory.jsonl`
- 10 个 MP4 视频：`videos/isaaclab_tactile_remote_0/eval_episode_0.mp4` 到 `eval_episode_9.mp4`

trajectory 文件：

- size: 4,342,449 bytes
- lines: 2,119

## 每集摘要

| ep | seed | steps | success | min eef-cube | final eef-cube | cube z max | final gripper cmd |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 0 | 1000 | 199 | true | 0.0355 | 0.2383 | 0.2092 | 0.964 |
| 1 | 1001 | 196 | true | 0.0349 | 0.2736 | 0.2135 | 0.953 |
| 2 | 1002 | 196 | true | 0.0426 | 0.2616 | 0.2038 | 0.990 |
| 3 | 1003 | 210 | true | 0.0360 | 0.2651 | 0.2065 | 1.009 |
| 4 | 1004 | 202 | true | 0.0409 | 0.2581 | 0.2035 | 1.020 |
| 5 | 1005 | 204 | true | 0.0330 | 0.2730 | 0.2159 | 0.926 |
| 6 | 1006 | 300 | false | 0.0583 | 0.2538 | 0.0233 | -0.938 |
| 7 | 1007 | 199 | true | 0.0381 | 0.2683 | 0.2102 | 0.935 |
| 8 | 1008 | 202 | true | 0.0383 | 0.2848 | 0.2156 | 0.965 |
| 9 | 1009 | 201 | true | 0.0373 | 0.2710 | 0.2116 | 1.040 |

## 失败 episode 初步判断

Episode 6 / seed 1006 失败不是 success 判定 plumbing 失效，而是实际未完成抓取/放置：

- 跑满 300 steps，`is_success=false`；
- cube z max 只有 `0.0233`，初始 z 约 `0.0220`，说明 cube 基本没有被抬起；
- min eef-cube distance 为 `0.0583`，明显大于成功集的 `0.0330–0.0426`；
- final gripper command 为 `-0.938`，与成功集末尾通常为正值不同；
- 对应视频最大：`eval_episode_6.mp4` 约 224 KiB，其它多为 132–156 KiB，也符合它跑满超时的表现。

因此，当前剩余失败模式更像是个别初始 pose / seed 下 approach 精度不足导致未有效接触或未成功夹持，而不是 camera refresh、通信、success 判定或 gripper command 全局符号错误。

## 运行中观察到的问题

1. Isaac server 在评估完成后的 cleanup 阶段出现 abort：

```text
ReferenceError: weakly-referenced object no longer exists
... tiled_camera.py __del__
```

该错误发生在 client 已经完成 10 episodes、写出 `eval_info.json`、`trajectory.jsonl` 和全部视频之后。它不影响本次评估结果，但说明 server 退出阶段仍有 Isaac camera/tiled camera 析构问题。

2. server 日志仍出现：

```text
Ill-formed SdfPath </World/envs/env_.*/Cube>
```

该 warning 在每个 seed 附近重复出现，但没有阻止评估完成。建议后续修正 telemetry 或 USD path 查询中使用 regex-like prim path 的代码，避免噪声和潜在隐藏问题。

3. `avg_sum_reward` 和 `avg_max_reward` 均为 0.0，但 success 正常记录。这说明当前 eval 主要依赖 termination success，而非 reward 数值；报告时应使用 `pc_success` 和 `successes` 列表作为主指标。

## 建议下一步

优先做针对 episode 6 / seed 1006 的失败复现和 pose 定向补强：

1. 用同一 checkpoint 单独跑 seed 1006 的 1-episode dense trajectory/video；
2. 读取 episode 6 初始 cube/basket pose；
3. 以该 pose 附近采集 3–5 条 targeted raw HDF5；
4. 用 `--drop-terminal-frame` 转换为独立 clean dataset；
5. 与当前 targeted5 merged 数据做小批再微调，验证是否能把 9/10 提升到 10/10 或提高多 seed 稳定性。

建议命令方向：

```bash
# 复现唯一失败 seed，保存更细的 trajectory/video
# 注意不要覆盖本次 n10 输出，使用新的 EVAL_OUT。
```
