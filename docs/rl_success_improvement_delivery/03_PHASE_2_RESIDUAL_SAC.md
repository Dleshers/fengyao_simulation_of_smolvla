# Phase 2：Residual SAC 模型与训练实现

## 1. 模型结构

### Torque encoder

```text
输入:  [B,30,7]
GRU:   hidden_size=64, num_layers=1, causal
输出:  最后隐藏状态 [B,64]
```

不直接复用冻结 SmolVLA 的 16 维 torque token，以便残差策略有足够容量学习时序变化；但必须做“去掉时序辅助损失”的消融。

### Actor

```text
concat(state[12], torque_latent[64], base_action[6], previous_action[6])
→ LayerNorm
→ Linear(256), SiLU
→ Linear(256), SiLU
→ mean[3], log_std[3]
```

`log_std` 限制在 `[-5,1]`，经 reparameterization 和 tanh 得到有界 residual。

### 非对称双 Critic

Critic 输入：

```text
actor feature
privileged [dx,dy,z,grasp_drift,torque_excursion,episode_progress]
residual_action[3]
```

使用两个独立 Q 网络及对应 target network，每个为三层 `256` 单元 MLP。

## 2. SAC 目标

Critic：

```text
y = r + gamma * (1-done) * [min(Q1_target,Q2_target) - alpha*log_pi]
L_Q = MSE(Q1,y) + MSE(Q2,y)
```

Actor：

```text
L_actor = E[alpha*log_pi - min(Q1,Q2)] + lambda_BC*L_BC
```

BC 锚定：

```text
delta_oracle = clip(oracle_action - base_action, residual_bounds)
L_BC = ||actor_mean - delta_oracle||^2
```

`lambda_BC` 在前 20k 环境步从 `1.0` 线性降到 `0.1`。

温度参数自动优化，目标熵为 `-3`。

## 3. 时序辅助损失

共享 torque GRU 增加一个二分类头，区分：

- 原始因果顺序窗口；
- 同一窗口内随机时间置换或完全反转。

```text
L_total_actor = L_actor + 0.05 * L_order
```

辅助标签只用于训练。正式实验必须包含 `L_order=0` 的 E5 消融，确认收益来自哪里。

## 4. 初始超参数

```yaml
gamma: 0.99
target_tau: 0.005
actor_lr: 0.0003
critic_lr: 0.0003
alpha_lr: 0.0003
batch_size: 256
replay_capacity: 500000
target_entropy: -3
updates_per_env_step: 1
actor_update_interval: 2
gradient_clip_norm: 10
delta_xy_max: 0.05
delta_z_max: 0.04
```

最终 residual 和安全动作范围应由 Phase 1 标定结果覆盖配置文件，不能默默使用默认值。

## 5. Replay Schema

Replay transition 至少保存：

```text
actor_observation
next_actor_observation
privileged_critic_state
next_privileged_critic_state
base_action
residual_action
executed_action
reward_total
reward_components
terminated
truncated
snapshot_id
episode_id
step_index
torque_mode
```

图像不必写入 replay：在线 rollout 时由冻结 SmolVLA 计算并保存 `base_action`。但每条 transition 必须保留来源 snapshot/episode，便于按轨迹抽样和审计。

## 6. 建议新增文件

```text
experiment/rl/residual_actor_critic.py
experiment/rl/residual_torque_policy.py
experiment/rl/replay_buffer.py
experiment/rl/train_residual_sac.py
experiment/rl/eval_residual_policy.py
experiment/rl/tests/test_residual_bounds.py
experiment/rl/tests/test_frozen_base.py
experiment/rl/tests/test_sac_losses.py
experiment/rl/tests/test_counterfactual_torque_modes.py
```

## 7. Plumbing Gate

在实际训练前必须通过：

1. residual 权重全零时，输出与基础 SmolVLA 一致；
2. base 模型所有参数 `requires_grad=false`，反向后 `.grad is None`；
3. Critic 可以看到特权字段，Actor 无法访问；
4. zero/shuffle 同时改变 base 与 residual 的 torque 输入；
5. replay 保存和加载后逐字段一致；
6. 100 个随机 batch 的 actor/critic/alpha loss 均有限；
7. 16 episode smoke 中没有系统性动作饱和或安全层持续触发。
