# 2026-07-07 baseline 闭环失败原因逐步排查

## 范围

本文件记录本地对 pure SmolVLA baseline 0% 闭环成功率的逐项排查。当前只做只读/小型离线推理与代码链路核验；未重新训练、未覆盖数据/模型、未启动批量闭环评估。

## 已排除或暂不支持的原因

### 1. baseline checkpoint 或 processor 损坏

本地 checkpoint 可加载，policy 可实例化，postprocessor 可把 action 恢复到训练集尺度。

已保存的离线审计位于：

- `experiment_results/2026-07-07_baseline_local_inference_audit/summary.json`
- `experiment_results/2026-07-07_baseline_local_inference_audit/samples.csv`

600 个训练集抽样帧上的结果：

- mean batch forward loss：`0.022345`
- action MAE by dim：`[0.01063, 0.02219, 0.01365, 0.00994, 0.01642, 0.02066, 0.02684, 0.00904]`
- gripper sign/binary accuracy：`100%`
- gripper MAE：`0.00904`

结论：当前没有证据表明上传的 baseline checkpoint、normalizer 或 postprocessor 损坏。

### 2. 训练集 action 不是 joint absolute target

采集脚本使用 IK-relative teacher rollout，但写入训练集的 policy label 是：

`[robot.data.joint_pos_target[:7], gripper_cmd]`

转换脚本明确声明：

- state：`[arm_joint_pos(7), gripper_qpos(2)]`
- action：`[arm_joint_pos_target_abs(7), gripper_cmd(1)]`

eval 的 joint TacEx 配置将前 7 维解释为 absolute joint target：

- `JointPositionActionCfg(scale=1.0, use_default_offset=False)`

结论：采集、转换、训练、eval joint action 的 7 维 arm 语义是自洽的。

### 3. gripper 符号翻转

IsaacLab `BinaryJointPositionAction` 的约定是：

- 正值 / `+1`：open
- 负值 / `-1`：close

采集 state machine 也是：

- `1.0`：open
- `-1.0`：close

训练集 gripper command 分布：

- `-1.0`：26,761 frames
- `+1.0`：14,515 frames

本地离线推理对 gripper sign 的准确率为 `100%`。

结论：目前没有证据支持 gripper 符号翻转。少量预测值会略超出 `±1`，但 IsaacLab gripper action 按符号二值化，通常不应导致 0% success。

### 4. LeRobot joint-mode env postprocessor 改写 action

`IsaacLabTactilePolicyActionProcessorStep` 在 `control_mode="joint"` 时直接返回 transition，不做坐标变换、缩放或 gripper 符号变换。

结论：LeRobot 侧 joint-mode action 后处理不是当前主要嫌疑。

### 5. state shape / camera key 明显错配

eval 侧 observation processor 在 joint 模式下构造：

`observation.state = [arm_joint_pos(7), gripper_qpos(2)]`

这与训练集 `[9]` 一致。

`eval_pair.sh` 使用 rename map：

`rgb_table -> observation.images.camera1`

`rgb_wrist -> observation.images.camera2`

eval server 输出 `rgb_table/rgb_wrist`，LeRobot env processor 也会根据 `camera_keys="rgb_table,rgb_wrist"` 生成 `camera1/camera2`。

结论：从代码路径看，state shape 和 camera key 没有明显错配。

## 当前强嫌疑

### A. `n_action_steps=50` 导致长 open-loop action chunk

baseline 配置：

- `chunk_size=50`
- `n_action_steps=50`
- 数据集 fps：20 Hz

SmolVLA `select_action()` 行为：

1. action queue 为空时，从当前 observation 预测一个 action chunk；
2. 将前 `n_action_steps` 个 action 放入 queue；
3. 后续每个 env step 只从 queue `popleft()`；
4. queue 清空前不会重新基于新图像推理。

因此当前闭环 eval 实际是：

- 每次视觉观测生成 50 个动作；
- 在 20 Hz 控制下，约 2.5 秒才重新看一次图像并重规划。

这不是实现错误，但对 pick/grasp/place 任务是强风险：只要初始视觉定位、接近路径或 cube 接触产生很小偏差，后续 2.5 秒动作仍按旧观测执行，容易出现抓空、推开 cube、错过夹爪闭合窗口或搬运中掉落。

需要视频/trajectory 进一步验证的表现：

- 前几步接近方向看似合理，但随后偏离；
- gripper 在错误空间位置闭合；
- cube 被碰到/推开而非稳定夹住；
- 中段 transport 或 release 时明显失配。

建议下一步做一个诊断性单 episode 对照，而不是直接改正式实验超参数：

- 当前配置 `n_action_steps=50` 跑 1 episode，保存 trajectory；
- 临时诊断配置 `n_action_steps=1` 或更短 action horizon 跑 1 episode，保存 trajectory；
- 只比较失败模式，不把诊断配置作为正式实验结果，除非用户确认。

### B. 闭环视觉分布偏移

离线推理在 demonstration frame 上表现很好，但 closed-loop 0% 说明模型可能只在 teacher 轨迹附近可靠。一旦动作误差累积，图像与 state 进入训练集未覆盖区域，策略会快速失效。

支持证据：

- 本地离线 action/gripper 与训练标签高度一致；
- 失败发生在 closed-loop，而不是 checkpoint 加载或前向输出阶段。

仍需 trajectory/video 证据：

- eef/cube 距离随时间是否先下降后上升；
- cube 是否在 grasp 前被推离；
- joint target error 是否持续偏大；
- policy action 是否逐渐超出训练集 joint/state 分布。

### C. 本地 eval server 资产/初始化稳定性

本机已确认：

- IsaacLab env 中 `pyzmq==27.1.0`
- LeRobot env 中 `pyzmq==27.1.0`
- `TRAJECTORY_LOG` 参数链路存在

但本机 eval server smoke 曾在 Joint TacEx env 初始化/资产加载阶段长时间未到达 ZMQ listen。日志中出现过 unresolved Omniverse/simready material asset `physics_stone.usda` 相关 warning。

结论：pyzmq 已不是阻塞点；本地 Isaac/资产初始化稳定性仍需先用最小 server probe 证明。

### D. 缺少本地 20 个 eval 视频和 trajectory logs

仓库和本地 runtime 中未找到两组 baseline eval 的 20 个视频、`eval_info.json` 或完整 trajectory JSONL。因此目前无法直接回答：

- 机械臂是否接近 cube；
- gripper 是否闭合；
- cube 是否被抓起、推开、掉落；
- action 尺度或 gripper command 是否在闭环中异常。

这仍是失败模式分类的必需证据。

## 建议的逐步验证顺序

1. 先启动单 episode eval server probe，确认本地 Isaac server 能 listen、reset、step，并写入非空 `TRAJECTORY_LOG`。
2. 用正式 baseline 当前配置跑 1 个诊断 episode，保存视频和 JSONL trajectory。
3. 从 trajectory 中计算：
   - eef/cube 距离；
   - cube/basket 距离；
   - gripper command 与 gripper qpos；
   - joint target error；
   - action 每维 min/max 与训练集分位数对比。
4. 若视频/trajectory 显示 long chunk 漂移，再请求确认后做 `n_action_steps=1` 的单 episode 对照。
5. 只有在单 episode 通信和日志稳定后，再考虑恢复 10 episode 或更大评估。

## 当前判断

当前最可能的原因排序：

1. `n_action_steps=50` 带来的长 open-loop chunk，在闭环接触任务中放大小误差；
2. 视觉闭环分布偏移 / teacher forcing 到 closed-loop 的 covariate shift；
3. 本地/远程 eval 环境或资产初始化差异；
4. 尚未被 trajectory 证实的 gripper 执行时序问题。

当前较不可能：

- baseline checkpoint 损坏；
- normalizer/postprocessor 缺失；
- action 8D 定义错误；
- gripper `+1/-1` 符号翻转；
- LeRobot joint-mode 后处理错误。

