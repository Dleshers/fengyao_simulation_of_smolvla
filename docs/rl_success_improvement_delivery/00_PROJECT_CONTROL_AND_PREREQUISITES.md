# Phase 前置：项目控制与执行约束

## 1. 仓库与工作树保护

真正的 Git 仓库为：

```text
/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla
```

执行 agent 必须先运行只读检查：

```bash
git status --short --branch
git branch --show-current
git remote -v
```

当前仓库可能包含用户已有的修改和未跟踪文件。禁止使用：

```text
git reset --hard
git checkout -- <path>
git clean -fd
```

所有 RL 新代码优先写入 `experiment/rl/`，所有新结果写入 `experiment_results/rl_success_improvement_v1/`。若必须修改已有 SmolVLA override 或 evaluator，应先记录原文件 SHA-256，并把修改限制在可审计的小补丁中。

## 2. 允许和禁止的信息

Actor 部署时允许输入：

- 两路 RGB，经冻结 SmolVLA 间接体现在基础动作中；
- 12 维机器人状态；
- 动作执行前的 `30×7` 有符号力矩窗口；
- 冻结基础策略动作；
- 上一步实际执行动作。

Actor 禁止输入：

- peg/hole 真值位置或相对位姿；
- 成功标志；
- 未来力矩；
- oracle 动作；
- 数据收集阶段标签或 load/sector 真值标签。

Critic、奖励函数和数据准入审计在仿真训练时可以使用特权几何信息，但这些字段必须与 Actor 输入构造完全分离，并由单元测试检查。

## 3. 统一成功与安全定义

严格成功：

```text
xy_error < 2.5 mm
-2 mm <= relative_depth <= 1 mm
连续保持 10 个控制步
```

安全失败：

```text
ejection:       relative_depth > 40 mm
pass-through:   relative_depth < -10 mm
grasp drift:    anchor drift > 3 mm
```

正式训练和评估不得自行改变这些定义。更严格的亚毫米对齐指标可以附加报告，但不能替换主终点。

## 4. 统一控制协议

- 控制频率约 15 Hz；
- `n_action_steps=1`，每执行一个动作后重新观测；
- 所有分支共享同一个初始仿真快照；
- 第一帧 RGB、机器人状态和基础力矩窗口必须相同；
- 快照恢复最大绝对误差不超过 `1e-6`；
- 正式分支使用相同 flow-noise 清单和相同推理采样数；
- zero/shuffle 干预必须同时作用于 torque SmolVLA 与 residual Actor。

## 5. 资源分工

建议：

- RTX 4090 或其他支持 Vulkan/RTX 的 GPU：Isaac Sim RGB rollout、快照生成和闭环评估；
- A100：不依赖可视化的离线网络训练或蒸馏；
- 若在同一 RTX 上完成全部训练，应先做 5k 环境步计时，基于实测吞吐量调整预算，而不是跳过门禁。

## 6. 全局停止条件

出现任一情况立即停止当前阶段：

1. Actor 输入中出现特权字段；
2. 首帧或快照一致性检查失败；
3. reward 出现 NaN/Inf 或奖励分项不守恒；
4. 训练成功率提升但安全失败显著增加；
5. validation 连续三个检查点无改善；
6. test manifest 被用于调参或 checkpoint 选择；
7. 发现训练/测试源 episode 或 snapshot hash 重叠。
