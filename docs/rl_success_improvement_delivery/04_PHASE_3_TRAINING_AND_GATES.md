# Phase 3：训练顺序与逐级门禁

## 1. 总原则

训练仅使用 Phase 0 的 train manifest；validation 仅用于 checkpoint 选择；test 在模型和超参数冻结前不得运行。

基础 SmolVLA 训练期间保持 `eval()` 和完全冻结。训练时每个 episode 随机选择基础 flow-noise seed，减少 residual 对单一基础动作样本的过拟合；validation 和 test 使用冻结的共同噪声清单。

## 2. Gate A：Oracle Residual BC Warm-up

从 train snapshot 恢复物理状态，使用已有恢复 oracle 生成有界动作：

```text
delta_oracle = clip(a_oracle - a_base)
```

仅训练 Actor 5,000 gradient steps，Critic 尚不参与决策。

通过条件：

- held-out train states 的 residual 方向与 oracle 正相关；
- 动作饱和率低于 5%；
- 16-pair validation smoke 中安全指标不劣于 `base+shield`；
- zero residual 回归测试继续通过。

## 3. Gate B：SAC Plumbing Smoke

运行单 seed、5k 环境步：

```text
warm replay transitions: 1000–2000
environment steps: 5000
checkpoint: 1000, 2500, 5000
validation smoke: 16 pairs
```

检查：

- Q target、actor loss、alpha 均无发散；
- replay reward 分布与 reward components 一致；
- terminal success 的平均 return 高于安全失败；
- residual 不是持续输出边界值；
- 不出现高频横向换向。

## 4. Gate C：奖励函数决策实验

用同一 seed、相同训练快照和预算比较：

| 实验 | 奖励 |
|---|---|
| E2 | 仅 sparse terminal + time penalty |
| E3 | 势函数 dense reward，不含安全/振荡惩罚 |
| E4 | 完整 dense safe reward |

每组训练 10k 环境步，在同一 96-pair validation 子集评估。

选择顺序必须是：

1. 先排除安全不劣失败的配置；
2. 在安全配置中最大化严格成功率；
3. 成功率接近时选择成功步数更短者。

不能仅根据累计 reward 选择配置。

## 5. Gate D：30k Decision Run

使用 E4 最佳配置，从相同 BC warm-up 起点重新训练，而不是继续奖励消融的 10k 权重。

```text
training seed: 1000
max environment steps: 30000
validation frequency: 5000
save: 5k, 10k, 15k, 20k, 25k, 30k
```

三次 validation 无改善时提前停止。最佳 checkpoint 采用以下字典序：

```text
safety_noninferior
→ lower_95ci_success_delta
→ strict_success_rate
→ median_success_steps
```

进入多 seed 正式训练的最低条件：

- 相对 `base+shield` 成功率为正；
- 没有增加弹出、穿透或抓取漂移；
- 两个载荷带均非负；
- 至少 6/8 方向非劣；
- original 优于同 checkpoint 的 zero 或 shuffle 中至少一个。

## 6. Gate E：三 Seed 正式训练

固定全部超参数后训练：

```text
seeds: 1000, 1001, 1002
max steps per seed: 50000
evaluation interval: 5000
early stop patience: 3 validations
```

每个 seed 单独选择 checkpoint，但选择规则完全相同。不得从 test 表现挑选 seed 或 checkpoint。

## 7. 阶段交付物

```text
experiment_results/rl_success_improvement_v1/phase_3/
  reward_ablation/
  seed_1000/
  seed_1001/
  seed_1002/
  checkpoint_selection.json
  training_curves.json
  REPORT.md
  METRICS.json
  RUN_MANIFEST.json
```

报告必须包含环境步数、wall-clock、GPU、显存峰值、replay 规模、所有损失曲线和每个 validation checkpoint 的安全/成功指标。
