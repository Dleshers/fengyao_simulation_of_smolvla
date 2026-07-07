# 2026-07-07 Gripper LSTM / SmolVLA 实验结果分析

## 记录日期

- 文档日期：2026-07-07
- 实验主要运行日期：2026-07-06 至 2026-07-07
- 运行位置：UCL 远程 headless IsaacLab / LeRobot workspace

## 实验目标

本轮实验的目标是完成 Franka pick-place 任务上的数据采集、LeRobot 数据集转换、纯 SmolVLA baseline 训练、baseline 闭环评估，并为后续 gripper torque LSTM / action expert 分支训练做准备。

## 工况说明

### 数据采集工况

- 任务：Franka pick-place basket tactile / visual / torque 数据采集
- 仿真环境：IsaacLab-Tactile headless rendering
- 背景模式：fixed
- 背景纹理：`small_empty_house_4k.hdr`
- 视觉输入：启用相机，224x224 图像
- 数据格式：
  - 原始 HDF5：`datasets/franka_pickplace_joint_torque_w30_20260706T130538Z/data.hdf5`
  - LeRobot 格式：`datasets/franka_pickplace_joint_visual_torque_w30_v1`
- torque 窗口：30 steps
- 采集规模：200 episodes，41,276 frames
- 数据集 fps：20
- action schema：8D joint action，包含 7D arm joint target 与 1D gripper command
- state schema：9D joint state，包含 7D arm joint position 与 2D gripper qpos

### baseline 训练工况

- 模型：pure SmolVLA baseline
- base model：`lerobot/smolvla_base`
- dataset：`Dleshers/franka-pickplace-joint-visual-torque-w30-v1`
- seed：1000
- training steps：50,000
- batch size：8
- optimizer / scheduler：cosine decay with warmup
- 输出目录：`gripper_lstm_experiments/baseline_smolvla_50000_seed1000`
- 最终 checkpoint：`checkpoints/050000`

### baseline 闭环评估工况

- 评估环境：IsaacLab tactile remote eval server
- 评估输入：视觉闭环策略评估
- seed：1000
- 评估 episode 数：
  - `eval_visual_seed1000`：10 episodes
  - `eval_visual_n5_seed1000`：10 episodes
- 输出：
  - `gripper_lstm_experiments/eval_visual_seed1000/eval_info.json`
  - `gripper_lstm_experiments/eval_visual_n5_seed1000/eval_info.json`
  - 共生成 20 个评估视频

### torque-LSTM 训练工况

- 模型：SmolVLA + gripper torque LSTM action expert
- dataset：同 baseline 数据集
- seed：1000
- 计划 training steps：50,000
- 实际进度：训练启动并运行至 step 5,000，随后在 checkpoint 保存阶段失败
- 输出目录：`gripper_lstm_experiments/torque_lstm_smolvla_50000_seed1000`

## 已完成结果

### 1. 数据集已完成并通过基本审计

原始 HDF5 数据集已完成：

- demos：200
- total_steps：41,276
- torque range：`[-57.5694, 19.1751]`
- SHA256 已生成

LeRobot 数据集已生成：

- total_episodes：200
- total_frames：41,276
- total_tasks：1
- features 包含：
  - `observation.state`：9D
  - `action`：8D
  - `observation.images.camera1`：224x224 RGB
  - `observation.images.camera2`：224x224 RGB
  - `observation.tactile.force_grid`：`2 x 10 x 12 x 3`
  - `observation.gripper_torque`：`30 x 1`

结论：数据采集与转换链路已经跑通，可支撑 baseline 与 torque-conditioned 分支训练。

### 2. 纯 SmolVLA baseline 已训练完成

训练日志显示 baseline 成功完成 50,000 steps：

- 最后 checkpoint：`050000`
- 结束标记：`PASS: 50000-step pure SmolVLA baseline completed`
- 训练末期 loss 大约在 `0.010 - 0.012` 区间

最终 checkpoint 已上传到 Hugging Face 私有仓库：

- repo：`Dleshers/smolvla-franka-pickplace-baseline-50k-seed1000`
- commit：`a526d99c4d40aa8a8632d8c3e93831891468f6bf`
- pipeline tag：robotics
- model artifacts 包含 `model.safetensors`、config、processor 配置和 train config

结论：baseline 训练与模型归档已完成。

### 3. baseline 闭环评估已完成，但任务成功率为 0%

两组视觉闭环评估结果一致：

| 评估目录 | episodes | avg_sum_reward | avg_max_reward | success rate |
| --- | ---: | ---: | ---: | ---: |
| `eval_visual_seed1000` | 10 | 0.0 | 0.0 | 0% |
| `eval_visual_n5_seed1000` | 10 | 0.0 | 0.0 | 0% |

环境日志显示 reward manager 当前没有 active reward terms，评估指标主要依赖 termination / success 条件。因此 `avg_reward = 0.0` 本身不一定能单独说明策略质量；更关键的是 `successes` 全部为 `false`。

结论：baseline 虽然可以完成训练和闭环 rollout，但在当前 IsaacLab remote visual 工况下没有完成成功 episode。后续需要优先分析视频、动作尺度、观测字段映射和 gripper command 行为。

## 未完成或失败项

### 1. torque-LSTM 分支训练未完成

torque-LSTM 训练已启动，训练过程本身能前向/反向运行到 step 5,000，但 checkpoint 保存失败。

失败点：

- 位置：step 5,000 checkpoint
- 错误类型：`safetensors` shared tensor / storage 保存错误
- 关键报错：`found no suitable name to keep for saving amongst: {'model.torque_lstm.lstm.weight_ih_l0'}`

初步判断：

- 训练计算链路基本可运行；
- 问题集中在 checkpoint serialization；
- 可能与 LSTM 参数 view / shared storage / tied tensor 检测有关；
- 需要调整保存逻辑或确保 torque LSTM 参数在 state dict 中不触发 `safetensors.torch.save_model` 的 shared tensor 检查。

### 2. `eval_visual_n10_seed1000` 未形成完整结果

目录 `gripper_lstm_experiments/eval_visual_n10_seed1000` 存在并包含视频目录，但未发现 `eval_info.json`。因此该组不能计入完成评估。

### 3. 最新 eval server 日志存在 pyzmq 缺失/进程终止记录

`logs/eval_visual_server.log` 中最新尾部记录显示 eval server 启动后遇到：

- `ModuleNotFoundError: No module named 'zmq'`
- 随后进程被 kill

这条日志可能对应后续单独启动 server 的失败尝试；已完成的 eval result 以两个 `eval_info.json` 为准。后续如果继续评估，需要先确认 IsaacLab conda 环境内 `pyzmq` 已安装且 eval server 能稳定启动。

## 结果分析

### baseline 训练 loss 收敛不等价于闭环成功

baseline 在离线训练中 loss 降到较低范围，但闭环 success rate 仍为 0%。这说明当前主要瓶颈不一定是 supervised loss，而可能在以下位置：

1. 观测字段映射  
   训练数据中的图像/state/action 字段与 remote eval server 返回字段必须严格一致。任何 camera key、state shape、normalizer 或 processor 映射偏差都可能导致策略闭环失效。

2. 动作尺度与 gripper command  
   数据 schema 中 action 是 8D，其中 gripper command 是 1D。若闭环 eval 中 gripper command 的尺度、符号、绝对/相对定义与训练集不一致，策略可能无法稳定抓取。

3. 环境 reward 设置  
   当前评估环境 reward manager 没有 active reward terms，因此不能通过 reward 曲线诊断进展，只能依赖 success flag、视频和 trajectory log。

4. 视觉域差异  
   采集工况使用 fixed background，评估也应严格匹配。如果 camera pose、rendering pipeline、background 或 observation preprocessing 有差异，视觉策略可能无法泛化。

### torque-LSTM 路线仍有价值，但当前阻塞在保存机制

torque-LSTM 分支能跑到 step 5,000，说明模型注册、dataset feature、forward path 和 optimizer 基本已接通。当前失败发生在保存 checkpoint 阶段，不是典型的训练数值崩溃。

因此下一步不应直接重跑 50k，而应先做一个最小复现：

- 使用很短 steps，例如 10-100 steps；
- 强制触发 checkpoint save；
- 验证 `model.torque_lstm.lstm.weight_ih_l0` 是否能被正常 clone / contiguous / independent storage 保存；
- 修复后再恢复完整 50k 训练。

## 建议下一步

1. 先修复 torque-LSTM checkpoint 保存问题，再跑完整 50k。
2. 对 baseline 的 20 个评估视频做人工检查，确认失败模式：
   - 是否能接近物体；
   - gripper 是否闭合；
   - cube 是否被碰撞、推出、抓起或掉落；
   - arm 是否动作幅度异常。
3. 对比训练 dataset action 分布与 eval trajectory 中 action 分布，重点检查 gripper command。
4. 在 eval server 中增加更明确的 rollout telemetry，例如 cube pose、eef pose、gripper command、success term 状态。
5. 重新确认 `pyzmq` 在 IsaacLab eval 环境中可用，再继续跑 `n10` 或 torque-LSTM 评估。

## 当前结论

截至 2026-07-07：

- 数据采集、数据转换、baseline 50k 训练、baseline Hugging Face 上传已完成；
- baseline 闭环视觉评估已完成两组，每组 10 episodes，但成功率均为 0%；
- torque-LSTM 训练链路已初步跑通，但 checkpoint 保存失败，尚未得到完整 50k 模型；
- 后续工作应优先修复 serialization，再结合视频和 trajectory 分析 baseline 失败原因。
