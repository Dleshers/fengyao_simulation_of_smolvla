# Phase 4：正式评估、消融与统计

## 1. 正式方法矩阵

在冻结的 384-pair test manifest 上运行：

| ID | 方法 | 目的 |
|---|---|---|
| B0 | Visual 10k | 视觉基线 |
| B1 | Torque 10k original | 当前主基线 |
| B2 | Torque 20k original | 单纯增加 BC 步数基线 |
| B3 | Torque 10k + shield | 分离安全层收益 |
| E1 | Residual BC + shield | 分离新增 oracle/模型容量收益 |
| E2 | Sparse SAC + shield | 稀疏奖励基线 |
| E3 | Dense SAC，无安全奖励 | 奖励消融 |
| E4 | Dense safe Residual SAC | 主方法 |
| E5 | E4 无 torque-order auxiliary | 时序辅助消融 |

若资源有限，E2/E3 可以只在 validation 报告，不进入正式 test；B0、B1、B3、E1、E4、E5 及 E4 的 torque counterfactual 必须保留。

## 2. Torque 因果干预

E4 使用同一组权重运行：

```text
rl_original
rl_zero
rl_shuffle
```

干预同时作用于：

1. torque SmolVLA 的 `30×7` 输入；
2. residual torque GRU 的 `30×7` 输入。

RGB、state、快照、flow noise、SAC deterministic mean、动作安全层和所有其他变量必须相同。

## 3. 测试协议

先执行 16-pair identity smoke。只有以下条件全部通过才可运行 384 对：

- `paired_initial_observation_identical=true`；
- restore error `<=1e-6`；
- 所有分支第一帧 RGB/state/base torque hash 一致；
- flow noise 清单一致；
- branch order 在不同 pair 间轮换，避免固定顺序偏差；
- 任一分支没有加载训练或 validation snapshot。

正式评估使用：

```text
n_action_steps=1
inference_samples=3
deterministic residual mean
max_branch_steps=240
strict_hold_steps=10
```

## 4. 指标

主要指标：

- 严格插入成功率；
- E4 相对 B1、B3 的配对成功率差；
- E4 original 相对 zero、shuffle 的配对差。

安全指标：

- ejection；
- pass-through；
- grasp drift failure；
- collision-tail termination；
- blocked downward push 数量。

效率与行为指标：

- time-to-success；
- minimum XY error；
- 首次 strict step；
- 横向动作换向次数；
- residual 饱和率；
- safety shield 触发率。

## 5. 分层统计

报告整体结果以及：

```text
8 direction sectors
2 load bands
3 initial XY bands
2 snapshot sources
```

二元成功结果使用配对 bootstrap 95% CI 和 McNemar 精确检验。bootstrap 必须按 snapshot/pair 重采样，不得按 step 重采样。建议至少 10,000 次 bootstrap。

多 seed 结果既要报告每个 seed，也要报告汇总；不能只展示最好 seed。

## 6. 工程通过标准

E4 必须同时满足：

1. 相对 B1 或 B3 严格成功率提高至少 5 个百分点；
2. 配对 bootstrap 95% CI 排除 0；
3. 弹出率不高于 1%，且不高于 B3；
4. pass-through、grasp drift 均安全非劣；
5. 成功时间不比 B1 慢超过 10%；
6. 两个载荷带中均为正向收益；
7. 至少 6/8 个方向不劣；
8. 三个训练 seed 中至少两个满足正向成功率差。

## 7. 论文级时序力矩结论

若要声称“时序力矩产生因果增益”，继续沿用项目既有的更严格门槛：

- original 分别领先 zero 和 shuffle 至少 15 个百分点；
- 两个配对 bootstrap 95% CI 均排除 0；
- 两个载荷带均存在增益；
- 至少 6/8 个方向覆盖增益；
- 不增加弹出、穿透或抓取漂移。

工程成功不自动等于时序因果结论通过，报告中必须分开表述。
