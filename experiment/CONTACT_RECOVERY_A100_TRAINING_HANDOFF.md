# 接触恢复数据与 A100 训练交接

## 当前状态

**可开始首轮训练。** 最终接触恢复集已平衡为 8 个方向各 8 条，并通过物理质量审计。必须使用下方指定的最终文件与 SHA-256。

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
