# Phase 1：奖励函数与 RL 环境

## 1. MDP 定义

一个 RL step 对应一个 Factory 控制步，约 15 Hz。基础 SmolVLA 每步重新推理，只执行动作块第一个动作。

Actor 观测：

```text
observation.state                float32[12]
observation.gripper_torque       float32[30,7]
base_action                      float32[6]
previous_executed_action         float32[6]
```

Residual Actor 输出：

```text
delta_translation_action         float32[3]
```

执行动作：

```text
a_exec = safety_shield(a_base + [delta_xyz, 0, 0, 0])
```

## 2. 几何变量

奖励函数使用仿真真值，但这些变量不得进入 Actor：

```text
dxy = ||held_xy - fixed_xy||，单位 mm
z   = held_z - fixed_z，单位 mm
drift = ||current_anchor - initial_anchor||，单位 mm
q = ||torque_t - median(pre_contact_torque)||_2
```

## 3. 势函数奖励

先定义横向门控：

```text
g(dxy) = sigmoid((0.6 - dxy) / 0.15)
```

势函数：

```text
Phi(s) =
    2 * exp(-dxy / 0.5)
  + 2 * g(dxy) * exp(-max(z - 1, 0) / 5)
```

进度奖励：

```text
r_progress = 5 * (gamma * Phi(s_next) - Phi(s))
gamma = 0.99
```

该设计使未对齐时的主要目标是减小 `dxy`；只有横向接近后，下降才获得明显正奖励。

## 4. 事件和安全奖励

```text
r_strict_step      = +0.2，每个 strict 步
r_success_terminal = +20，连续 strict 10 步
r_safety_terminal  = -20，弹出/穿透/抓取漂移
r_blocked_push     = -1.0
r_oscillation      = -0.5
r_time             = -0.01
```

Blocked push：

```text
executed_z_action < -0.08
且 |z_next - z| < 0.1 mm
```

Oscillation：

```text
dot(a_xy_t, a_xy_previous) < 0
且 dxy 没有至少改善 0.02 mm
```

## 5. 力矩与动作惩罚

从 Phase 0 的安全成功轨迹中统计全局 `q95` 和 `q99.5`：

```text
force_penalty =
2 * ReLU((q - q95) / max(q99.5 - q95, epsilon))^2
```

惩罚裁剪到 `[0,4]`。超过单独标定的碰撞尾部阈值时立即终止。

动作正则：

```text
r_residual = -0.02 * ||delta / delta_max||^2
r_smooth   = -0.02 * normalized(||a_exec - a_previous||^2)
```

完整奖励为所有分项之和。日志必须保存每一项，不能只保存总奖励。

## 6. 安全层

安全层属于独立实验因素，不能把其收益归因于 SAC。它至少执行：

1. residual XYZ 范围裁剪；
2. 最终动作裁剪到成功示教动作的逐维 `[q0.5, q99.5]`，同时保留全局 `[-0.35,0.35]`；
3. 当力矩处于高尾部且下降无进展时，将负 Z 动作截为 0，允许卸载；
4. 旋转维完全沿用基础策略；
5. residual 仅在已定义的近孔恢复 episode 中启用。

## 7. 建议新增文件

```text
experiment/rl/__init__.py
experiment/rl/factory_rl_reward.py
experiment/rl/factory_snapshot_env.py
experiment/rl/action_safety_shield.py
experiment/rl/configs/residual_sac_v1.yaml
experiment/rl/tests/test_reward.py
experiment/rl/tests/test_safety_shield.py
experiment/rl/tests/test_actor_observation_contract.py
```

## 8. 测试与通过条件

必须验证：

- 向孔中心移动时进度奖励增加；
- 未对齐下压不能得到净正奖励；
- 连续 10 步 strict 才产生终局成功；
- 弹出、穿透、漂移立即终止；
- reward 分项之和与总奖励数值一致；
- Actor batch 中不存在 `held_pos`、`fixed_pos`、`xy_error`、`success` 等字段；
- 零残差加关闭安全层时与基础策略动作逐元素相同；
- 所有输出均为有限值。

阶段通过前，只允许运行不超过 16 个 episode 的 smoke。
