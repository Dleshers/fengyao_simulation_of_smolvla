# SmolVLA 强化学习成功率提升：分阶段交付包

## 1. 目标

本交付包用于在当前 Factory peg-in-hole 项目基础上继续进行强化学习，目标是在不破坏视觉粗定位能力的前提下：

1. 提高近孔接触恢复后的严格插入成功率；
2. 降低弹出、穿透、抓取漂移和孔口振荡；
3. 缩短成功恢复所需时间；
4. 验证策略确实利用了原始、因果、有时间顺序的 `30×7` 关节力矩窗口；
5. 最终根据需要把小型 RL 残差策略蒸馏回 SmolVLA。

推荐主路线为：

```text
冻结 torque SmolVLA 10k
        │
        ├── 数据支持范围安全层
        │
        └── 近孔三维平移 Residual SAC
                    │
                    ├── 原始/置零/乱序力矩因果评估
                    │
                    └── 可选：优势加权 flow-matching 蒸馏
```

不建议第一步直接对完整 SmolVLA 做 PPO。当前策略通过 flow-matching 多步去噪产生动作，没有直接提供稳定可用的动作对数概率；同时现有策略成功率已经较高，全模型在线更新更容易发生灾难性遗忘。

## 2. 当前冻结基线

当前最重要的同快照 64 对结果为：

| 策略 | 严格成功率 | 弹出 | 平均成功步数 |
|---|---:|---:|---:|
| Visual 10k | 44/64 = 68.8% | 1 | 43.0 |
| Torque-zero 10k | 48/64 = 75.0% | 0 | 61.3 |
| Torque-shuffle 10k | 54/64 = 84.4% | 0 | 58.0 |
| Torque-original 10k | 56/64 = 87.5% | 3 | 53.0 |

RL 的工程目标为：将新冻结测试集上的成功率提高到约 `92%–95%`，同时把弹出率压到不高于 `1%`，并保持穿透和抓取漂移安全非劣。

## 3. 文档索引

| 文档 | 阶段 | 主要交付内容 |
|---|---|---|
| `00_PROJECT_CONTROL_AND_PREREQUISITES.md` | 项目控制 | 权限边界、工作树保护、统一定义、停止规则 |
| `01_PHASE_0_BASELINE_AND_MANIFEST.md` | Phase 0 | 基线冻结、快照分层、train/validation/test manifest |
| `02_PHASE_1_REWARD_AND_ENVIRONMENT.md` | Phase 1 | MDP、奖励函数、终止条件、安全层与测试 |
| `03_PHASE_2_RESIDUAL_SAC.md` | Phase 2 | Actor/Critic、SAC 损失、残差动作与实现接口 |
| `04_PHASE_3_TRAINING_AND_GATES.md` | Phase 3 | BC warm-up、SAC 训练、检查点与逐级门禁 |
| `05_PHASE_4_FORMAL_EVALUATION.md` | Phase 4 | 消融矩阵、同快照测试、统计和通过标准 |
| `06_PHASE_5_DISTILLATION_AND_RELEASE.md` | Phase 5 | 优势加权蒸馏、回归验证与发布材料 |
| `07_LUNA_MAX_EXECUTION_PLAYBOOK.md` | 执行手册 | 供 GPT-5.6 Luna Max 使用的逐阶段任务模板 |

## 4. 阶段依赖

每个阶段必须生成机器可读 JSON 和人工可读 Markdown 报告。上一阶段未通过时，不得进入下一阶段：

```text
Phase 0 基线与数据清单
  → Phase 1 奖励与环境单元测试
  → Phase 2 零残差等价性和 SAC plumbing
  → Phase 3 训练与 validation 决策
  → Phase 4 冻结 test 正式评估
  → Phase 5 可选蒸馏与发布
```

## 5. 统一结果目录

新实验不得覆盖历史结果。统一使用：

```text
experiment_results/rl_success_improvement_v1/
  phase_0/
  phase_1/
  phase_2/
  phase_3/
  phase_4/
  phase_5/
```

每个阶段至少包含：

```text
REPORT.md
METRICS.json
RUN_MANIFEST.json
commands.sh.txt
```

`commands.sh.txt` 仅记录实际执行过的命令，不应包含访问令牌或其他秘密。
