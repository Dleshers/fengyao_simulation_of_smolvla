# 接触恢复数据与 A100 训练交接

## 当前状态

**当前任务是使用修正后的 current-action loss 重新运行 5k 门禁，不是正式长训。** `contact_recovery_reactive_phase5_gate_20260813` 的旧损失权重已判定 no-go，禁止续训；`balanced64/reactive-phase5` 仅用于修复验证。A100 必须从相同 official SmolVLA base 分别重训 visual 与 torque-original，并以本文末尾“current-action loss 修复后的 A100 重训门禁”为最新执行规范。只有该门禁通过后，才允许使用 v4 的 320 条接触恢复 + 80 条常规插入数据进行正式训练。

## 目的

本轮实验先比较纯视觉与原始力矩 LSTM：视觉应完成粗定位；力矩仅在孔口阻挡后的卸载、重对齐与再插入阶段提供额外历史信息。

## 数据准入

- 视觉定位主体：Hub 路径 `visual_oneway_v2/peg_insert_demos.hdf5`（120 条；SHA-256: `bc979a1366ab6268ea5433c417ca736ddd0b8b1eb911c9d7ba71b8318357312d`）。
- 接触恢复补充：Hub 路径 `contact_recovery_native_v1_balanced64/peg_insert_demos.hdf5`（64 条，8方向各8条；SHA-256: `325571bc0048617a8f9d5bbe7df65b8628d1210d1fee5a4f9833f94a2739f5ae`）。
- 训练前必须运行两个审计脚本；不通过时不得训练或替换数据。
- 原始 HDF5 不提交 GitHub。请上传至 Hugging Face 私有数据集 `Dleshers/factory-peg-insert-contact-recovery-native-v1-gateb64`，A100 从该仓库下载并校验 SHA-256。

## 下载与校验

```bash
HF_REPO="Dleshers/factory-peg-insert-contact-recovery-native-v1-gateb64"
RAW_ROOT="$PWD/persistent/raw_hdf5"
hf download "$HF_REPO" --repo-type dataset --local-dir "$RAW_ROOT"
sha256sum "$RAW_ROOT/visual_oneway_v2/peg_insert_demos.hdf5"
sha256sum "$RAW_ROOT/contact_recovery_native_v1_balanced64/peg_insert_demos.hdf5"
```

仅接受 GitHub 交接文档中记录的最终 SHA-256；不得使用早期的非平衡 `gateb64` 文件训练。

## A100 环境

1. 克隆 GitHub 仓库并检出记录本次交接的提交。
2. 用项目的 IsaacLab/LeRobot 环境或等价 CUDA 环境安装依赖；训练不需要 Isaac Sim 图形渲染，仿真评估才需要 RTX/Vulkan 环境。
3. 下载两份原始 HDF5 至 `persistent/raw_hdf5/`，运行审计：

```bash
ISAAC_PY="$ROOT/.conda/isaaclab/bin/python"
$ISAAC_PY experiment/audit_factory_peg_insert_visual_oneway_v2.py --input "$VISUAL_RAW" --output-dir "$VISUAL_AUDIT"
$ISAAC_PY experiment/audit_factory_peg_insert_contact_recovery_native_v1.py --input "$CONTACT_RAW" --output-dir "$CONTACT_AUDIT" --expected-demos 64
```

## 转换与公平训练

最终发布后，先使用数据清单筛选平衡的 64 条接触轨迹。视觉主体可完整转换；接触集必须使用 `--policy-label-only --torque-dim 7`，这样恢复监督帧保留前 30 帧真实力矩历史。

转换器必须保留接触轨迹的完整历史，但只对 `is_policy_label=true` 的恢复帧计算行为克隆损失；不能把接触建立阶段当作专家恢复标签。训练集按轨迹（接触集按 `pair_id`）划分，两个模型使用同一划分、训练步数、学习率、数据顺序种子与检查点选择准则。

首轮仅运行两组：

| 组 | 输入 |
| --- | --- |
| visual | 两路 RGB + 12D proprioception；无力矩分支 |
| torque-original | 相同视觉/proprioception + 连续 30×7 signed joint-torque LSTM |

旧的 `trained_lstm_weights/torque_16d_encoder.pt` 仅接受 1D 力矩范数，不能直接加载到本轮 30×7 signed-torque LSTM；本轮应新初始化相同容量的 7D LSTM。之后的零力矩与因果打乱力矩消融必须复用同一检查点、评估状态和随机种子。

## 评估规则

在未见轨迹、固定的近孔阻挡初始状态上同时报告严格恢复率、重对齐率、夹持漂移、时间到成功。视觉组必须先有有效粗定位；否则不得把力矩组差异解释为触觉收益。

## 2026-08-13 强制更新：修复后重训要求（以本节为准）

### 已确认的问题与边界

- 已评估的旧 visual / torque-original checkpoint 是**诊断基线**，不是最终可比较模型：其训练时 SmolVLA 读取了不存在的 `actions_id_pad`，而数据集实际提供的是 `action_is_pad`。在 `chunk_size=50` 下，padding 动作会错误参与损失（视觉原始集约 17.93%，接触恢复标签序列约 47.69%），足以同时损害两组闭环行为。
- 仿真评估器也已修正：闭环执行必须使用 `--n-action-steps 1`，每一步重新观察 RGB 与力矩；不得使用 checkpoint 中遗留的 `n_action_steps=50`。该错误会令两组连续执行 50 个过期动作。
- 这两个共同因素已能解释旧评估的低共同成功率。因此在修复后两组的配对、分层评估完成前，**不得声称触觉无效或有效**。

### 固定代码版本与预检

```bash
git fetch origin main
git checkout main  # 当前交接文档及 padding/评估修复均在 main

# 将仓库中已修复的 override 安装到实际训练环境；LEROBOT_ROOT 是该环境的 lerobot 源码根目录。
cp remote_handoff_gripper_lstm/lerobot_overrides/configuration_smolvla.py \
   "$LEROBOT_ROOT/src/lerobot/policies/smolvla/configuration_smolvla.py"
cp remote_handoff_gripper_lstm/lerobot_overrides/modeling_smolvla.py \
   "$LEROBOT_ROOT/src/lerobot/policies/smolvla/modeling_smolvla.py"
grep -n 'actions_is_pad = batch.get' "$LEROBOT_ROOT/src/lerobot/policies/smolvla/modeling_smolvla.py"
```

上面最后一行必须显示 `batch.get("action_is_pad")`。训练启动前还必须通过仓库回归测试（按实际环境设置 `ISAAC_PY`）：

```bash
PYTHONPATH="$LEROBOT_ROOT/src" "$ISAAC_PY" -m pytest -q \
  remote_handoff_gripper_lstm/lerobot_overrides/test_smolvla_torque_lstm.py
```

### 不可变的数据划分

先下载并校验上文两个原始 HDF5；之后**先生成并保存** `TRAIN_SPLIT_MANIFEST.json`，再物化 train HDF5 和转换数据。禁止对完整 120+64 条数据直接转换后随机按帧划分。

| 数据来源 | train | holdout | 划分约束 |
| --- | ---: | ---: | --- |
| visual_oneway_v2 | 100 demos | 20 demos | 按 demo 划分，固定随机种子 |
| contact_recovery_native_v1_balanced64 | 56 demos | 8 demos | 按 `pair_id` 分组；同一 pair 不可跨 split；holdout 覆盖 8 个扇区 |

清单至少应记录：Git commit、两个源文件 SHA-256、每条 demo 名称、来源（visual/contact）、contact 的 `pair_id` 与 sector、split、帧数、转换参数及随机种子。两个训练臂必须读取**同一份 train manifest**；所有最终评估仅读取 holdout manifest 和明确列出的未见随机种子。

### 训练顺序与公平性

1. 仅从相同官方 SmolVLA base 新建两个 run：`visual` 和 `torque-original`；禁止从旧 checkpoint 续训，也禁止把 `trained_lstm_weights/torque_16d_encoder.pt` 加载进 7D 分支。
2. 两组使用相同 train manifest、样本/数据顺序种子、batch size、optimizer、学习率计划、总步数、checkpoint 选择准则和 RGB/proprioception 预处理。唯一差异是 torque-original 接收连续 **30×7 signed joint torque** 及其新初始化的 LSTM。
3. contact 数据转换仍必须使用 `--policy-label-only --torque-dim 7`；保留接触前 30 帧真实历史，但仅对 `is_policy_label=true` 的恢复帧监督动作。
4. 先各跑一个短门禁 run（建议 5k steps），确认训练/验证 loss 正常、padding loss 为零且能加载推理；通过后再从官方 base 各自完整训练（建议至少 20k steps，并按验证集选择 checkpoint）。不要把短 run checkpoint 用作正式结果。

每个 run 应上传 `config.json`、训练命令、commit、manifest 副本、所有 checkpoint、train/valid metrics、随机种子和环境包版本到 Hugging Face；Git 仅提交小型代码、清单和报告，不提交 HDF5 或大权重。

### 修复后的评估门禁

评估环境必须使用本仓库修复后的 `experiment/eval_factory_peg_insert_native_contact_takeover.py`，显式传入 `--n-action-steps 1` 和固定 seed；先做 8 个配对 holdout 冒烟回合（同一初始状态分别跑 visual 与 torque-original），再做每组至少 32 个配对回合。

分别按横向初始误差、接触/非接触初始化、sector、是否先进入近孔区域分层，报告：strict insertion success、alignment recovery、contact-to-success recovery、首次接触后成功率、夹持漂移、完成步数，以及逐 seed 的 visual/torque 配对结果。只有在视觉组已可靠进入近孔区域、且两组共同失败不再由定位或执行错误主导时，才可把 torque-original 的配对增益解释为触觉带来的恢复收益。


## 2026-08-13 5k gate common-failure correction

The downloaded padfix 5k checkpoints passed loading and padding plumbing, but
both arms still fail the current-action recovery gate. Exact replay excluded
camera, state, torque-window, action-scale and controller mismatches. The
legacy balanced64 labels contain a fixed-timer phase-4 prefix that is not
observable by the policy; useful recenter corrections occur mainly later in
the 50-step action chunk. Do not change formal evaluation away from
`n_action_steps=1` and do not warm-start from these checkpoints.

For a corrected short retraining gate, convert the contact HDF5 with
`--policy-label-only --policy-phase-min 5 --torque-dim 7`. The resulting local
audit set has 64 episodes and 2,390 frames; 64/64 first XY labels point toward
the hole centre (mean cosine 0.9976, minimum 0.9683). Before closed-loop
evaluation, run `experiment/audit_native_contact_action_chunks.py` and require
reliable first-action direction. Full evidence and thresholds are in
`experiment_results/COMMON_FAILURE_ROOT_CAUSE_AND_FIX_20260813.md`.

Phase-5 exact-frame audit confirms that the old weights remain underfit after
removing the timer ambiguity: visual is 6/8 sectors at mean cosine 0.392,
while torque-original is 5/8 at mean 0.296. Both arms must therefore restart
from the common base on the corrected labels; do not reuse either 5k
checkpoint.

## 2026-08-13 v4 正式因果实验：本节覆盖本文此前的正式训练安排

完整规范见同目录的 [`CONTACT_RECOVERY_V4_DATASET_DESIGN.md`](CONTACT_RECOVERY_V4_DATASET_DESIGN.md)。A100 agent 必须先阅读该文件；它是验证“触觉时序信息具有正向作用”的正式实验设计，而不是可选参考。

### 当前允许与禁止事项

- `contact_recovery_native_v1_balanced64` 以及本文此前的 120+64 数据，只允许用于安装验证、转换审计、`action_is_pad` 修复验证和至多 5k/2k 步的管线门禁；**不得**将其完整训练结果作为正式触觉结论。
- 在 v4 的 Gate A--D 通过、正式数据及其哈希/manifest 上传到 Hub 前，A100 不得发起正式 visual / torque-original 长训，也不得用旧 checkpoint 续训。
- v4 正式训练数据到位后，旧的“visual 100/20、contact 56/8”划分不再适用；必须按 v4 的 `pair_id` manifest 划分，并保留独立、从未参与训练或模型选择的 120-state 配对评估 manifest。

### A100 的正式前置门禁

1. 按本文“固定代码版本与预检”安装 `action_is_pad` 修复，并通过回归测试；评估器固定 `--n-action-steps 1`。
2. 在 v4 Gate A 中完成物理接触几何/载荷校准；Gate B 的 64 条 smoke 数据须覆盖 8 方向 × 2 偏差带 × 2 载荷带 × 2 重复。
3. Gate C 必须先以按 `pair_id` 分组的轻量 probe 证明：30×7 原始力矩能预测纠偏方向，而冻结接触 RGB + proprioception 不足以提供同等信息；不满足阈值时先修数据生成器，不能靠增加训练步数解决。
4. Gate D 仅训练 torque 组 2k 步，确认 original torque 相对 zero/causal-shuffle 会改变纠偏动作且闭环不系统性弹出；通过后才开始 Gate E 正式数据采集与训练。

### v4 正式训练和结论要求

- 正式集为 320 条接触恢复轨迹 + 80 条常规严格插入轨迹；接触后因果训练视图使用冻结的接触 RGB，但原始 HDF5 同时保留 live RGB 供部署视图后续验证。
- policy 输入仅可含 RGB、12D proprioception 和可选 30×7 signed torque window；peg/hole 真值、接触标签、恢复阶段和 oracle 信息仅作审计元数据，绝不能输入模型。
- visual 与 torque-original 必须从相同修复后的官方 base 启动，使用同一 manifest、采样/训练种子、优化器和 checkpoint 选择标准；唯一变量是原始力矩 LSTM suffix token。
- 所有模型在同一 120 个保存初始状态上配对评估：visual、torque-original、同一 torque checkpoint 的 zero torque 与 causal-shuffle torque。结论同时报告严格恢复、重对齐、弹出/夹持漂移与时间到成功。
- 只有 original torque 同时优于 zero 和 causal-shuffle 至少 15 个百分点，两个配对 bootstrap 95% 区间均不跨零，并且增益覆盖两个偏差带及至少 6/8 方向，才可声称触觉带来正向恢复收益。

## 2026-08-13 current-action loss 修复后的 A100 重训门禁（最新执行要求）

### 本轮决策

`contact_recovery_reactive_phase5_gate_20260813` 的旧损失 5k 权重不得续训，也不得启动 v4 正式长训。32 条方向均衡离线审计中，visual 首动作平均 XY 余弦为 `0.341`（21/32 正向），torque-original 为 `0.146`（19/32 正向）；相同 seed 的原生物理接触闭环均为 0/2 对齐、0/2 严格插入，torque 两回合还出现越界。与此同时，24 状态反事实表明 original torque 平均纠偏余弦 `0.456`，zero 为 `-0.198`，causal-shuffle 为 `0.005`，说明 LSTM 已使用触觉，但首个闭环动作监督仍不足。

### 代码修复

SmolVLA 行为克隆损失现在具有两个明确性质：

1. `action_is_pad=true` 的时间步不进入分子或分母；每个样本按其有效的加权动作元素归一化，短 chunk 不再因固定 50 步分母获得更小梯度。
2. 配置项 `action_loss_first_step_weight` 对 chunk index 0 加权；默认 `1.0` 保持其他任务兼容，本门禁 visual 与 torque 均固定为 `5.0`。在 50 步完整 chunk 中，首动作占时间损失的比例由 2% 提升到约 9.26%，同时保留后续轨迹监督。

A100 必须从本提交的 GitHub `main` 安装两份 override：

```bash
REPO=/path/to/fengyao_simulation_of_smolvla
LEROBOT_ROOT=/path/to/lerobot-tactile
PY=/path/to/python

cp "$REPO/remote_handoff_gripper_lstm/lerobot_overrides/configuration_smolvla.py" \
  "$LEROBOT_ROOT/src/lerobot/policies/smolvla/configuration_smolvla.py"
cp "$REPO/remote_handoff_gripper_lstm/lerobot_overrides/modeling_smolvla.py" \
  "$LEROBOT_ROOT/src/lerobot/policies/smolvla/modeling_smolvla.py"

PYTHONPATH="$LEROBOT_ROOT/src" "$PY" -m pytest -q \
  "$REPO/remote_handoff_gripper_lstm/lerobot_overrides/test_smolvla_torque_lstm.py"
```

预期结果为 `8 passed`。还必须保存以下审计输出：

```bash
grep -n "action_loss_first_step_weight" \
  "$LEROBOT_ROOT/src/lerobot/policies/smolvla/"{configuration_smolvla.py,modeling_smolvla.py}
git -C "$REPO" rev-parse HEAD
```

### 训练数据和初始化

- 仅使用修正后的 `Dleshers/factory-peg-insert-contact-recovery-v1-7d-reactive-phase5-train` 门禁数据；其来源转换必须包含 `--policy-label-only --policy-phase-min 5 --torque-dim 7`。
- 两组都从同一 padding-fixed official SmolVLA base **重新初始化**；不得从任何旧 5k 权重 resume。
- 固定 `seed=1000`、`batch_size=32`、相同 sampler、optimizer、学习率计划、保存步和总步数。唯一模型差异仍是 torque-original 的 30x7 signed-torque LSTM。
- 两组都显式传入 `--policy.action_loss_first_step_weight=5.0`。不得只给 torque 加权，否则对照不公平。

训练命令沿用 A100 上一次成功运行的命令，只替换新的官方 base 输出目录并追加：

```bash
COMMON_ARGS=(
  --dataset.repo_id=Dleshers/factory-peg-insert-contact-recovery-v1-7d-reactive-phase5-train
  --batch_size=32
  --steps=5000
  --seed=1000
  --policy.action_loss_first_step_weight=5.0
)

# visual：use_torque_lstm=false
# torque-original：use_torque_lstm=true, torque_window_size=30,
# torque_input_dim=7, torque_lstm_hidden_dim=32, torque_lstm_output_dim=16,
# torque_lstm_num_layers=1, train_torque_lstm=true, torque_lstm_weights_path=""
```

在 step 2000 与 step 5000 都保存 checkpoint。此处 5k 仍只是门禁，不是正式结果；若 2k 已明显失败，也保留其指标以判断学习曲线，不得从旧 checkpoint 混接。

### 强制评估与 go/no-go

每个 2k/5k checkpoint 先运行 `experiment/audit_native_contact_action_chunks.py --phase-min 5 --demos-per-sector 4`，再以 `n_action_steps=1` 运行相同 seed 的 2+2 原生接触闭环。5k 只有同时满足以下条件才通过：

- visual：8 方向中至少 6 个首动作方向为正，方向均衡 32 样本平均首动作 XY 余弦 >= `0.35`；
- torque-original：至少 7/8 方向为正，32 样本平均首动作 XY 余弦 >= `0.55`；
- 相同状态下 original torque 的纠偏方向同时优于 zero 和 causal-shuffle；
- visual/torque 的 2+2 均无系统性弹出、负深度越界或夹持漂移，并至少表现出重对齐能力。

判定顺序：

1. 全部通过：才允许开始 v4 的 320 接触恢复 + 80 常规插入正式训练。
2. visual 和 torque 都失败：继续检查共同训练目标、归一化或动作执行，不扩充数据。
3. visual 通过但 torque 失败且反事实仍为正：先调整首动作权重（只可对两组一起改）或 torque 采样均衡，再做短门禁。
4. 修复损失后仅固定 sector/载荷失败：才按失败 cell 定向补充配对轨迹；禁止无差别扩增同类数据。
5. torque 对 zero/shuffle 无差异：返回 v4 Gate C 修改接触载荷与观测设计，不进入正式长训。

所有 checkpoint、`train_config.json`、逐 step loss、32 样本离线 JSON、2+2 闭环 JSON、Git commit 和测试日志上传到 Hugging Face。
