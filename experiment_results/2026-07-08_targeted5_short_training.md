# 2026-07-08 Targeted5 短程验证训练记录

## 结论

已完成短程验证训练准备，并成功完成一次 visual-only baseline 微调短训：

- 训练数据：原 200 集 LeRobot 数据集去除未使用的 `observation.tactile.force_grid` 后，与 5 个 targeted clean 小批数据集合并。
- 合并后数据集：215 episodes，44,338 frames。
- 训练分支：SmolVLA visual-only baseline，`use_tactile=false`，`use_torque_lstm=false`。
- 训练步数：5,000 steps。
- 保存频率：1,000 steps。
- 结果：001000、002000、003000、004000、005000 checkpoint 全部成功保存，训练正常结束，退出码 0。

这说明 targeted5 数据合并后的 schema、normalizer/preprocessor 保存、baseline checkpoint 加载、训练循环和 safetensors 保存链路均可用于后续小规模消融验证。

## 输入资产

### Baseline checkpoint

本地路径：

```text
_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000
```

关键配置：

- `type=smolvla`
- `use_tactile=false`
- `use_torque_lstm=false`
- `model.safetensors` 大小约 906.7 MB

### Targeted clean 数据集

5 个高频失败 pose 的小批 targeted clean 数据集此前已完成采集、转换与审计：

| dataset | episodes | frames |
|---|---:|---:|
| `targeted5_center_cluster_w30_n3_clean_v1` | 3 | 613 |
| `targeted5_right_center_w30_n3_clean_v1` | 3 | 610 |
| `targeted5_right_upper_w30_n3_clean_v1` | 3 | 609 |
| `targeted5_right_lower_w30_n3_clean_v1` | 3 | 610 |
| `targeted5_left_lower_w30_n3_clean_v1` | 3 | 620 |

## 数据集准备

### Schema 对齐

原官方本地数据集包含额外特征：

```text
observation.tactile.force_grid
```

targeted clean 数据集不包含该特征。由于本次验证训练明确使用 visual-only baseline，并设置：

```text
--policy.use_tactile=false
--policy.use_torque_lstm=false
```

因此用 LeRobot `remove_feature` 生成了 common-schema 官方数据副本，仅用于本次合并验证，不覆盖原始官方数据集。

common-schema 数据保留特征：

- `observation.state`
- `action`
- `observation.images.camera1`
- `observation.images.camera2`
- `observation.gripper_torque`
- `timestamp`
- `frame_index`
- `episode_index`
- `index`
- `task_index`

### 本地合并脚本

新增脚本：

```text
experiment/merge_lerobot_datasets_local.py
```

用途：

- 显式传入各 source dataset root；
- 调用 LeRobot lower-level `aggregate_datasets`；
- 默认 `HF_HUB_OFFLINE=1`；
- 输出存在时拒绝覆盖；
- 避免 `lerobot-edit-dataset --operation.type merge` 将 `--root` 解释为单个 dataset root 并触发 Hugging Face fallback。

### 合并输出

本地路径：

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean
```

合并结果：

```json
{
  "total_episodes": 215,
  "total_frames": 44338,
  "fps": 20,
  "features": [
    "action",
    "episode_index",
    "frame_index",
    "index",
    "observation.gripper_torque",
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.state",
    "task_index",
    "timestamp"
  ]
}
```

## 数据集审计

验证命令：

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python \
  _runtime/remote_handoff_gripper_lstm_work/experiment/validate_dataset.py \
  --repo-id franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean \
  --root _runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean \
  --window-size 30 \
  --samples 128 \
  --sequence-checks 2048
```

审计结果：

- 2,048 个 sequential torque windows 通过；
- `state`: `[9]`
- `action`: `[8]`
- `observation.images.camera1`: `[3,224,224]`
- `observation.images.camera2`: `[3,224,224]`
- `observation.gripper_torque`: `[30,1]`
- state range: `[-2.86092, 3.06616]`
- action range: `[-2.85657, 3.08995]`
- torque range: `[-48.2476, 3.47857]`

边界样本检查：

- idx 0: ep 0, frame 0
- idx 41275: ep 199, frame 202
- idx 41276: ep 200, frame 0
- idx 44337: ep 214, frame 197

## Smoke test

第一次 2-step smoke 使用默认 dataloader workers 时失败：

```text
OSError: AF_UNIX path too long
```

原因判断：

- `TMPDIR` 路径过长；
- DataLoader multiprocessing resource sharer 使用 AF_UNIX socket；
- 与数据内容或模型权重无关。

修正：

```bash
mkdir -p /tmp/svl
export TMPDIR=/tmp/svl
export TMP=/tmp/svl
export TEMP=/tmp/svl
```

并设置：

```text
--num_workers=0
```

修正后 2-step smoke 成功：

- step 1: loss `0.009`, grad `0.207`
- step 2: loss `0.045`, grad `1.580`
- 正常结束

## 5k 短程验证训练

输出目录：

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_ablation_visual_5000_seed1000
```

关键参数：

```text
dataset.repo_id=franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean
dataset.root=_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1_plus_targeted5_clean
policy.path=_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000
policy.device=cuda
policy.use_tactile=false
policy.use_torque_lstm=false
batch_size=8
steps=5000
num_workers=0
save_freq=1000
seed=1000
wandb.enable=false
```

训练日志摘要：

| step | loss | grad | lr | 状态 |
|---:|---:|---:|---:|---|
| 100 | 0.016 | 0.367 | 3.1e-05 | 正常 |
| 500 | 0.028 | 0.508 | 9.8e-05 | 正常 |
| 1000 | 0.029 | 0.510 | 9.2e-05 | checkpoint 成功 |
| 2000 | 0.023 | 0.428 | 6.8e-05 | checkpoint 成功 |
| 3000 | 0.017 | 0.361 | 3.8e-05 | checkpoint 成功 |
| 4000 | 0.014 | 0.280 | 1.3e-05 | checkpoint 成功 |
| 5000 | 0.014 | 0.259 | 2.5e-06 | checkpoint 成功，训练结束 |

训练结束日志：

```text
Checkpoint policy after step 5000
End of training
```

最终 checkpoint：

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_ablation_visual_5000_seed1000/checkpoints/005000/pretrained_model
```

最终 checkpoint 只读检查：

- `model.safetensors`: 906,712,520 bytes
- `config.json`: 3,432 bytes
- `train_config.json`: 8,135 bytes
- `training_state/training_step.json`: `{"step": 5000}`
- safetensors 可被 `safe_open(..., device="cpu")` 打开；
- tensor count: 500；
- `has_torque_lstm=false`，符合 visual-only baseline 验证目标。

## 注意事项

1. 本次不是 torque-LSTM 分支训练；它是 targeted5 数据对 visual baseline 的短程可训性验证。
2. common-schema 官方数据副本删除了未使用的 `observation.tactile.force_grid`，原官方数据集未覆盖。
3. `SmolVLAPolicy.from_pretrained()` 的独立 CPU 加载检查在沙盒网络受限时会尝试访问 Hugging Face 上的 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 配置文件并重试；该检查被中断。训练过程已经在 CUDA 环境中成功加载 baseline 并完成 5k steps。
4. 后续如要做完全离线加载检查，应确认 SmolVLM2 processor/config 相关文件均已完整缓存，或在代码侧显式传入本地 VLM 目录。

## 建议下一步

用最终 5k checkpoint 做小规模闭环评估，对比原 baseline 在同一 pose/seed 下的失败模式：

```bash
TRAJECTORY_LOG=1 \
POLICY_PATH=_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_ablation_visual_5000_seed1000/checkpoints/005000/pretrained_model \
EVAL_OUT=_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_ablation_visual_5k_eval_n10_seed1000_20260708 \
bash _runtime/remote_handoff_gripper_lstm_work/experiment/run_eval_server.sh
```

然后用相同 eval client 参数运行 10 episode，对比：

- approach distance；
- gripper close timing；
- cube lift / push-away / drop；
- success 判定；
- action 8D 分布与原 baseline 的差异。
