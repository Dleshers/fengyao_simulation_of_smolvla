# 毕业论文写作交接：任务介绍与方法论要点

> 用途：供另一个 AI agent 快速理解本项目并据此撰写毕业论文。本文只保留研究问题、技术路线和实验设计等关键信息；正式实验结果暂留空。
>
> 当前方法以 `experiment/` 中 2026-08-14 的最新设计和实际代码为准。仓库中早期的 1D 力矩、冻结外部 LSTM 权重和旧版 causal-v2 文档仅作历史记录，不应作为论文当前方法。

## 1. 研究任务与核心问题

- 任务：在 Isaac Sim / Isaac Lab 的 Factory 环境中，控制机械臂将圆柱销插入孔中。
- 主要研究问题：当视觉已经完成粗定位、销到达孔口或接触边缘后，带符号的关节力矩时间序列能否帮助策略识别接触状态、恢复横向偏差，并提高严格插入成功率？
- 核心假设：
  1. 视觉适合完成自由空间中的全局定位和接近孔口；
  2. 在遮挡、微小间隙和接触约束下，单帧图像难以完整表达接触方向与载荷变化；
  3. 具有时间顺序的 7D 关节力矩窗口可提供接触方向、载荷演化和恢复趋势；
  4. 将该时间信息注入 Action Expert，应能改善近孔接触后的闭环恢复，而不是替代视觉完成全局找孔。
- 论文应避免把研究问题写成“触觉是否在所有阶段都优于视觉”。更准确的表述是：**在视觉粗定位有效的前提下，时序力矩信息是否对接触阶段产生可验证的增益。**

## 2. 与参考文章及 π0/SmolVLA 范式的关系

- 参考文章：*End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy: VR Teleoperation Augmented by Autonomous Hand VLA Policy for Efficient Data Collection*，arXiv:2511.00139v2。
- 本项目是对其“VLA 动作生成 + 触觉/力觉信息参与控制”思路的迁移和简化，不是逐模块复刻。
- SmolVLA 与 π0 在方法范式上相似：均将视觉语言表征与 Action Expert 结合，并通过 flow matching 生成连续动作；但两者不是完全相同的网络实现，论文中应写“采用与 π0 相近的 VLM + Action Expert + flow-matching 范式”。
- 参考工作偏向空间化触觉表征；本项目不使用触觉图像或力阵列，改用最近 30 帧、7 个关节的带符号力矩时间序列，经 LSTM 压缩为一个条件 token。
- 该 LSTM token 并不替代视觉空间特征，而是作为紧凑的时间接触线索，补充到 Action Expert 的条件序列中。

## 3. 仿真任务、输入与输出

- 仿真平台：Isaac Sim 4.5、IsaacLab-Tactile、Factory peg-in-hole。
- 推荐运行环境：Ubuntu 22.04、Python 3.10、PyTorch 2.7 + CUDA 12.8；仿真和视觉评估使用支持 Vulkan/RTX 的 GPU，A100 仅用于训练。
- 视觉输入：桌面相机和侧面相机两路 RGB；采集分辨率 84×84，转换为 LeRobot 数据时处理为 224×224。
- 机器人状态：12 维，包含 9 个关节位置和 3 维指尖中点位置。
- 力矩输入：按时间从旧到新排列的 `30 × 7` 有符号关节力矩窗口；必须是动作执行前可获得的历史，禁止未来信息泄漏。
- 动作输出：6 维末端增量位姿，由 Factory 控制器执行。
- 控制频率：约 15 Hz；在线评估每次只执行预测动作块的第一个动作，然后重新观测和规划，即 `n_action_steps=1`。
- 语言指令：`Insert the peg into the hole`。
- 当前严格成功判据：横向误差小于 2.5 mm，且插入深度处于规定区间，并满足连续保持要求。正式论文须同时报告更严格的接触恢复指标，避免宽松成功阈值掩盖差异。
- 几何背景：当前圆孔直径约 8.100 mm、销直径约 7.986 mm，径向间隙约 0.057 mm，因此最终插入对亚毫米对齐非常敏感。

## 4. 模型方法

### 4.1 纯视觉基线

- 使用官方 SmolVLA 预训练基座初始化。
- 输入两路 RGB、机器人状态和语言指令。
- 不启用力矩 LSTM：`use_torque_lstm=false`。

### 4.2 时序力矩模型

- 视觉、状态、语言和 Action Expert 与纯视觉组完全相同。
- 力矩编码器：一层 LSTM，`input_dim=7`、`hidden_dim=32`，取最后隐藏状态后通过线性层投影到 16 维。
- 16 维向量经过 LayerNorm 和线性投影，映射到 Action Expert 隐藏维度，形成一个 torque token。
- torque token 插入 Action Expert suffix，并位于动作时间 token 之前；它不进入 VLM prefix。
- 每次策略调用只计算一次 torque token，并在本次 flow-matching 去噪过程中复用。
- 当前正式方案从随机初始化开始端到端训练 LSTM：`train_torque_lstm=true`。
- 旧文件 `trained_lstm_weights/torque_16d_encoder.pt` 是早期 `30 × 1` 输入编码器，与当前 `30 × 7` 结构不兼容，不能用于本轮训练或作为当前论文方法。

### 4.3 唯一主要自变量

- 视觉组和力矩组应使用相同预训练基座、数据、样本顺序、训练种子、优化器、训练步数和 Action Expert 配置。
- 两组的主要结构差异只应是力矩组多出的 causal torque token。
- 因而两组之间的性能差异才可归因于时序力矩条件，而不是数据、初始化或训练预算差异。

## 5. 数据集设计

### 5.1 数据应覆盖的控制阶段

- 自由空间视觉接近：保证纯视觉模型能够可靠到达孔口附近。
- 缓慢接触：形成真实而连续的 30 帧力矩历史。
- 近孔受阻：覆盖不同方向、不同载荷的边缘接触和亚毫米失败状态。
- 恢复动作：先卸载，再横向回正，最后重新下插；必要时允许一次受控重试。
- 已成功插入、直接穿透、无有效接触、接触后漂移或只保留成功尾帧的轨迹均不得作为有效恢复示教。

### 5.2 当前决策数据集

- 基础集 `balanced64`：8 个方向扇区均衡覆盖的原生重置接触恢复轨迹。
- 定向补充 `hard16`：由当前策略实际访问并失败的 `<1 mm` 近孔状态产生，按 8 个方向扇区和 2 个载荷带进行平衡。
- 合并集 `hard80 = balanced64 + hard16`，用于短预算决策训练；它用于判断方法是否值得扩展，并不自动等同于最终论文规模的正式数据集。
- `hard16` 原始数据与 `hard80` LeRobot 数据应分别保存到文档指定的 Hugging Face 私有数据集。
- 若短预算实验显示稳定正向信号，再扩充更大规模训练集和独立留出评估集；扩充时应优先增加真实策略失败状态，而不是简单重复容易成功的轨迹。

### 5.3 数据采集与审计原则

- 采集从原生仿真重置开始，不允许通过直接写入最终姿态伪造接触历史。
- 力矩历史必须由真实连续物理交互产生。
- oracle 可用于恢复动作标签、几何审计和采集接受判定，但 oracle 位姿、孔真值或成功标志不得作为策略输入。
- 保存动作前的 RGB、机器人状态和力矩窗口，使训练时的观测—动作时序与在线评估一致。
- 审计实际接触偏差和载荷，不只审计命令值；检查 8 个方向扇区、载荷带、接触激励、恢复方向和严格成功保持。
- 训练/验证划分按轨迹或成对样本 ID 进行，禁止相邻帧跨集合造成泄漏。

## 6. 数据转换与训练方法

- HDF5 转 LeRobot 时使用：
  - `--torque-control original`
  - `--torque-dim 7`
  - `--policy-label-only`
  - `--policy-phase-min 5`
- `policy-phase-min=5` 用于剔除隐藏的定时前缀，只训练真正闭环恢复动作，避免模型学习共同的固定阶段行为。
- `action_is_pad` 必须正确传入损失函数并屏蔽动作块中的 padding；早期拼写错误会使两组共同学习无效补零动作。
- 当前动作损失对动作块第一个时间步赋予 5 倍权重，以匹配在线执行时只采用第一个动作的设置：

  `L = Σ(m_i w_i loss_i) / [D Σ(m_i w_i)]`，其中 `w_0=5`、其余为 1，`m_i` 为非 padding 掩码，`D` 为动作维数。

- 推荐训练流程：
  1. 两组分别从相同官方基座开始 2k smoke 训练，验证 loss、保存和推理链路；
  2. smoke 通过后，两组都重新从相同基座训练 10k，而不是从 2k 继续，以保持可比性；
  3. 保存 2k、5k、8k、10k 检查点；
  4. 根据配对闭环评估决定是否扩展到 20k/50k 及更大数据集。
- 当前约定：seed=1000、batch size=32；两组必须使用相同帧顺序和训练预算。

## 7. 评估方法与因果验证链路

### 7.1 为什么不能只比较总体成功率

- 总体成功率同时受视觉找孔、接近速度、控制器、夹持稳定和接触恢复影响。
- 若两组都在到达孔口前失败，触觉没有机会发挥作用，其潜在增益会被共同失败掩盖。
- 因此必须分层报告：到达近孔率、有效接触初始化率、近孔横向恢复率、严格插入成功率和恢复后保持率。

### 7.2 同状态配对闭环评估

- 视觉组和力矩组必须从同一仿真快照分叉，具有相同机器人、物体、相机、控制器和初始接触状态。
- 在同一进程中恢复快照，并审计状态恢复误差；独立启动、只使用相同随机种子不等价于严格配对。
- 分叉后的第一帧 RGB 应经过一致性检查，必要时向两分支重放完全相同的首帧；后续均使用实时渲染图像。
- 策略在距离孔约 3.5 mm 的近孔区域接管，以便比较接触恢复，而不是比较全局视觉搜索。

### 7.3 必需对照

- `visual`：没有 torque token 的架构基线。
- `torque-original`：使用真实、按时间排列的 7D 力矩窗口。
- `torque-zero`：使用同一个力矩检查点，但推理时将力矩窗口置零。
- `torque-shuffle`：使用同一个力矩检查点，但破坏时间顺序或配对关系。
- `zero/shuffle` 是关键因果干预：若 original 仅优于 visual，却不优于 zero/shuffle，不能证明模型真正利用了力矩时间信息。

### 7.4 指标与统计

- 主要指标：严格插入恢复成功率。
- 分层指标：近孔到达率、有效受扰初始化率、横向重新对齐率、卸载成功率、恢复后再次接触率、弹出率、夹持漂移率、成功所需时间。
- 按 8 个方向扇区、不同初始偏差和高/低载荷分别统计，避免总体平均值掩盖失败区域。
- 使用配对 bootstrap 计算成功率差的置信区间；二元配对结果可使用 McNemar 检验。
- 建议的正式触觉结论门槛：`torque-original` 同时优于 `torque-zero` 和 `torque-shuffle`，优势在两个载荷带中都存在，并覆盖至少 6/8 个方向扇区；最终阈值需在看正式结果前固定。

## 8. 主要有效性威胁

- 视觉组基础成功率过低：说明视觉粗定位或评估链路仍有共同问题，不能直接解释为触觉无效。
- 数据只含容易成功轨迹：模型没有学到真实失败状态下的恢复策略。
- 力矩窗口缺少真实时间连续性：LSTM 只能学习静态幅值，无法验证时间信息。
- 训练—评估时序不一致：例如训练使用动作后观测、评估使用动作前观测，或开放环执行过长。
- padding 污染和固定阶段前缀：会共同压低视觉与力矩模型的有效首动作质量。
- 独立仿真启动造成初态漂移：会破坏配对比较并放大 Isaac Sim 的非确定性。
- 只比较不同架构模型：无法排除额外参数量的影响，必须加入同检查点的 zero/shuffle 干预。
- 成功阈值过宽：可能把“到达孔附近”误判为“完成精密插入”。

## 9. 关键实现与文档索引

- 最新短预算训练交接：`experiment/ACTUAL_SUBMM_FAILURE_A100_TRAINING_HANDOFF_20260814.md`
- 接触恢复数据设计：`experiment/CONTACT_RECOVERY_V4_DATASET_DESIGN.md`
- 训练与评估要求：`experiment/REACTIVE_PHASE5_10K_TRAINING_HANDOFF_20260814.md`
- SmolVLA 配置：`remote_handoff_gripper_lstm/lerobot_overrides/configuration_smolvla.py`
- SmolVLA/LSTM/损失实现：`remote_handoff_gripper_lstm/lerobot_overrides/modeling_smolvla.py`
- hard16 采集：`experiment/collect_factory_peg_insert_policy_failure_recovery_v1.py`
- hard16 审计：`experiment/audit_factory_peg_insert_policy_failure_recovery_v1.py`
- hard80 合并：`experiment/materialize_factory_peg_insert_contact_recovery_v2.py`
- LeRobot 转换：`experiment/convert_factory_peg_insert_hdf5_to_lerobot.py`
- 同快照配对评估：`experiment/eval_factory_peg_insert_same_state_pair.py`
- 训练—评估链路审计：`experiment_results/TRAIN_EVAL_PIPELINE_AUDIT_20260813.md`
- 共同失败根因记录：`experiment_results/COMMON_FAILURE_ROOT_CAUSE_AND_FIX_20260813.md`
- 5k 诊断验证记录：`experiment_results/REACTIVE_PHASE5_FIRSTSTEPW5_5K_VALIDATION_20260814.md`

## 10. 论文结果部分占位

以下内容待正式实验结束后补充，撰写方法章节时不要根据中间日志预设结论：

| 内容 | 待填写信息 |
|---|---|
| 数据质量审计 | 轨迹数、扇区/载荷分布、接触激励、接受率、剔除原因 |
| 训练收敛 | visual/torque 的 loss 曲线、检查点选择依据、训练资源和时长 |
| 离线验证 | 首动作方向正确率、幅值误差、original/zero/shuffle 差异 |
| 闭环总体结果 | visual、torque-original、zero、shuffle 的严格成功率 |
| 分层结果 | 近孔、不同偏差、不同载荷、8 个扇区的恢复率与置信区间 |
| 配对统计 | 成功率差、bootstrap 置信区间、McNemar 检验 |
| 失败分析 | 视觉未到孔、未有效接触、卸载失败、对齐失败、插入失败 |
| 定性材料 | 同初态配对视频、力矩时间曲线、动作轨迹和典型成功/失败案例 |

## 11. 建议论文结构

1. 绪论：精密装配中的视觉局限、接触信息价值和研究问题。
2. 相关工作：VLA、π0/SmolVLA、模仿学习、视觉—触觉融合、peg-in-hole。
3. 系统与方法：Factory 环境、SmolVLA、7D 时序力矩 LSTM token、动作损失。
4. 数据与实验设计：恢复示教、hard-state 定向补充、质量审计和公平对照。
5. 实验结果：总体与分层结果、zero/shuffle 因果干预、配对统计。
6. 讨论：触觉发挥作用的阶段、失败模式、仿真到真实迁移限制。
7. 结论与展望。

## 12. 撰写时必须保持的表述边界

- 当前信号来源是机器人关节力矩时间序列，可称为“力矩/接触信息”或“触觉代理信号”；若称“触觉”，需在论文中先定义其含义。
- 不应声称复现了 π0 或参考文章的完整结构；应声称复现其核心研究思路，并在 SmolVLA 上实现时序力矩条件化 Action Expert。
- 不应仅凭 torque 模型优于 visual 就断言时序触觉有效；还需要 original 优于 zero/shuffle 的因果证据。
- 不应把旧 1D `torque_16d_encoder.pt` 写入当前方法；当前模型使用 7D 力矩窗口并端到端训练新的 LSTM。
- 在结果未完成前，只写研究假设、预注册式判据和实验设计，不提前写“证明了触觉有效”。
