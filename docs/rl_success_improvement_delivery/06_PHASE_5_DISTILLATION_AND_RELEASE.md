# Phase 5：优势加权蒸馏与发布

## 1. 何时执行

本阶段为可选阶段。只有 Residual SAC 的 Phase 4 工程标准通过后才执行。若部署允许保留“小型 residual Actor + 冻结 SmolVLA”，可以不蒸馏。

蒸馏目标是得到单体 torque SmolVLA，减少部署组件，同时尽量保留 RL 获得的安全和成功率增益。

## 2. 蒸馏数据

从 train/validation 而非 test 中生成：

- Residual SAC 严格成功轨迹；
- SAC 失败状态上的 oracle recovery；
- 原 hard80/hard128 成功示教，用于防止遗忘。

每一帧保存 SAC 实际执行动作 `a_exec`，并保持 pre-action RGB/state/torque 对齐。不得只保存成功尾部或把未来力矩写入当前窗口。

建议混合：

```text
50% 原始 hard80/hard128 示教
30% RL 严格成功轨迹
20% RL 失败状态 oracle recovery
```

按轨迹和 `(source,sector,load,xy_band)` 平衡抽样，不能让长 episode 按帧数支配训练。

## 3. 优势加权 Flow Matching

利用 SmolVLA 已有逐样本 loss：

```text
advantage_i = Q_i - V_i
weight_i = clip(exp(advantage_i / beta), 0.1, 10)
weight_i = weight_i / mean(weight_batch)
L_AWFM = sum(weight_i * flow_loss_i) / sum(weight_i)
```

初始 `beta=1.0`。必须记录实际权重的均值、标准差、p95、最大值和有效样本量；若大量样本被裁剪到 10，应提高 beta。

加入参考策略锚定：

```text
L_total = L_AWFM + 0.1 * L_reference
```

`L_reference` 使用原始示教和冻结 10k 策略在固定 noise/time 下的 flow velocity target，减少视觉粗定位与普通插入能力遗忘。

## 4. 训练配置

建议：

```text
initialize: frozen 10k torque checkpoint
train: Action Expert + torque LSTM/projection
freeze: vision encoder
learning rate: 1e-5
batch size: 32
steps: 5k and 10k
first-step weight: 5.0
action padding: action_is_pad
```

必须保留 unweighted 相同数据训练作为对照，区分“新增成功数据”与“优势权重”的贡献。

## 5. 回归验证

蒸馏模型依次通过：

1. 加载、normalizer、shape 和 `action_is_pad` 测试；
2. 原始普通插入 validation，检查粗定位没有明显退化；
3. 96-pair RL validation；
4. 模型、超参数全部冻结后，运行与 Phase 4 相同的 384-pair test；
5. original/zero/shuffle 同 checkpoint 干预。

若蒸馏模型低于 residual 系统超过 3 个百分点，默认保留 residual 系统，不以部署简洁为由隐藏性能退化。

## 6. 发布内容

```text
model checkpoint(s)
config.json
normalizer/unnormalizer
SHA256SUMS
training manifest
dataset revisions and hashes
evaluation manifest hash
formal metrics JSON
model card
reproduction commands
```

模型卡必须明确：

- Actor 实际使用哪些观测；
- Critic 使用了哪些仿真特权信息；
- 是否保留 residual 模块和安全层；
- test 集是否只运行一次；
- 工程成功与时序因果结论分别是否通过。
