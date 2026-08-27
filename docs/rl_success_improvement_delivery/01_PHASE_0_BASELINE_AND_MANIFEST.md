# Phase 0：基线冻结与快照 Manifest

## 1. 阶段目标

本阶段不训练模型。目标是冻结可复现基线、建立全新的 train/validation/test 快照清单，并避免继续在历史 formal64 上调参。

## 2. 冻结模型

主要基础策略：

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/models/
contact_recovery_actual_failure80_smoke_20260815/runs/
torque/checkpoints/010000/pretrained_model
```

参考基线：

```text
.../visual/checkpoints/010000/pretrained_model
.../torque/checkpoints/020000/pretrained_model
```

记录以下文件的 SHA-256：

- `model.safetensors`；
- `config.json`；
- normalizer/unnormalizer 文件；
- 数据集 `meta/info.json`；
- 当前 Git commit。

20k 结果使用过不同的 flow-noise 和快照初始化细节，只能作为参考。进入新实验后，10k、20k 与 RL 策略必须在同一新清单、同一 evaluator 参数下重跑。

## 3. 快照分层设计

构造以下 96 个单元：

```text
8 direction sectors
× 2 contact load bands
× 3 XY bands
× 2 snapshot sources
```

XY 带：

```text
0.05–0.2 mm
0.2–0.5 mm
0.5–1.0 mm
```

来源：

- `controlled_contact`：由受控原生物理接触产生；
- `policy_failure`：由冻结 10k 策略真实访问的受阻、过冲、反向振荡状态产生。

每个快照都必须来自原生物理过程。允许恢复实际访问过的快照，但不允许把物体直接写入成功姿态。

## 4. 数据规模

| 集合 | 每单元快照 | 总数 | 用途 |
|---|---:|---:|---|
| train | 8 | 768 | replay 采集和训练 |
| validation | 2 | 192 | checkpoint 选择和超参数决策 |
| test | 4 | 384 | 最终一次性正式评估 |

若资源不足，允许先做 smoke 子集：

```text
train: 每单元 1，共 96
validation: 每个 sector/load 1，共 16
test: 暂不运行
```

smoke 子集不能用于正式统计结论。

## 5. Manifest Schema

每个 manifest 行至少包含：

```json
{
  "snapshot_id": "...",
  "split": "train|validation|test",
  "seed": 0,
  "source": "controlled_contact|policy_failure",
  "sector": 0,
  "load_band": 0,
  "xy_band": "0.2_0.5mm",
  "initial_xy_error_m": 0.0003,
  "initial_depth_m": 0.02,
  "snapshot_sha256": "...",
  "first_rgb_sha256": {"camera1": "...", "camera2": "..."},
  "state_sha256": "...",
  "torque_window_sha256": "...",
  "source_policy_sha256": "...",
  "git_commit": "..."
}
```

## 6. 阶段测试

必须自动检查：

1. 96 个单元是否达到计划数量；
2. split 间 seed、source episode、snapshot hash 是否完全不重叠；
3. 快照恢复误差是否 `<=1e-6`；
4. 首帧 RGB/state/torque 是否在配对分支中一致；
5. 所有起点是否尚未严格成功；
6. 所有 `policy_failure` 是否确实具有 blocked、overshoot、flip 或 non-progress 证据；
7. test manifest 是否被标记为只读并记录 SHA-256。

## 7. 阶段交付物与通过条件

```text
experiment_results/rl_success_improvement_v1/phase_0/
  baseline_hashes.json
  train_manifest.jsonl
  validation_manifest.jsonl
  test_manifest.jsonl
  split_audit.json
  REPORT.md
  METRICS.json
  RUN_MANIFEST.json
```

通过条件：所有分层计数正确、无 split 泄漏、一致性审计全部通过。否则禁止实现或训练 RL。
