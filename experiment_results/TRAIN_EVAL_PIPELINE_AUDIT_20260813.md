# 训练集与评估流程联合审计（2026-08-13）

## 结论

目前不应立即补采训练数据。审计确认了两个先于补采/重训必须修复的问题：

- 正式 native-contact 评估继承 checkpoint 的 `n_action_steps=50`，每次观测后开环执行 50 步；这会同时削弱视觉闭环和力矩闭环。
- SmolVLA 训练读取了拼错的 `actions_id_pad`，而 LeRobot 实际产生 `action_is_pad`，导致越过 episode 末尾的重复动作仍被计入 loss。

第二项尤其严重：在 `chunk_size=50` 下，视觉主体约 17.9% 的动作槽位是本应屏蔽的 padding；接触恢复标签序列只有 44--58 帧，其 padding 槽位约占 47.7%。这会促使两组共同学习过量的末端/停滞动作，能够解释孔口附近共同失败率很高的现象。

## 已排除的评估问题

- 精确回放 `demo_00000`：12D state、30x7 torque window、oracle action 最大误差均为 0。
- 两路 84x84 相机平均绝对误差约为 0.139/255 与 0.108/255，不是有意义的视觉域偏移。
- 采集、转换与评估均为 pre-action 对齐；接管点已对齐到训练首标签帧。
- 数据频率 15 Hz 与环境 `0.0666667 s` step 一致。
- 大多数旧评估失败从未进入 2.5 mm xy 阈值，不是严格成功判据漏报。
- visual/torque 使用同一数据路径、50k steps、batch 8、seed 1000，归一化统计一致。

## 训练集审计

- Visual Oneway v2：120 条、16,396 帧；全部 native reset、pre-action、严格成功。
- Native Contact Recovery balanced64：64 条、14,614 原始帧；8 方向各 8 条；只选择策略标签后为 3,286 帧。
- SHA-256 与交接一致：visual `bc979a1366ab6268ea5433c417ca736ddd0b8b1eb911c9d7ba71b8318357312d`；contact `325571bc0048617a8f9d5bbe7df65b8628d1210d1fee5a4f9833f94a2739f5ae`。
- checkpoint normalizer 计数 16,516，与视觉留 20 条验证、接触分层留出验证的 100+56 划分量级一致。
- `episode_index.max=99` 不能解释为只训练了 100 条：LeRobot 聚合 stats 时不会给第二来源的 episode_index 统计加偏移。
- A100 最终 train/holdout episode 清单和物化数据未上传，仍是复现缺口；下次必须保存 split manifest。
- balanced64 的正式 format 原先不在转换器白名单，已修复。

## 旧结果的解释限制

旧 32+32、`n_action_steps=50`：visual aligned 10/32、strict 5/32；torque aligned 11/32、strict 7/32。配对为 both-success 2、visual-only 3、torque-only 5、both-fail 22。力矩有 +6.25 pp 趋势，但在评估开环与训练 padding 污染存在时，不能作为触觉结论。

## 修正后闭环准入结果

`n_action_steps=1`、未见 seed 20261300--20261303、native reset、物理孔口接触、30 帧真实 7D 力矩历史：

| arm | aligned | strict |
|---|---:|---:|
| visual | 1/4 (25%) | 1/4 (25%) |
| torque-original | 1/4 (25%) | 1/4 (25%) |

配对为 both-success 1、visual-only 0、torque-only 0、both-fail 3；共同成功发生在 seed 20261303。两组独立 Isaac 进程的 contact XY 最大差异约 0.31 mm，应在正式配对统计中继续记录，但远小于失败后的 6--25 mm 横向误差。

该 smoke 证明修正后的评估链路可完整运行，但没有证明当前 torque checkpoint 优于 visual。力矩失败回合出现持续负 Z 和横向过冲，和训练 padding 污染的机制相符。因此不得通过扩大旧权重评估样本来代替修复后重训。

## 已实施修复

- native-contact evaluator 默认 `n_action_steps=1`，显式固定环境 seed，并记录动作步数与轨迹诊断。
- SmolVLA 改为读取 `action_is_pad`。
- 转换器接受 balanced64 正式格式。

## 后续门禁顺序

1. 已完成 4+4 同 seed、逐步闭环准入评估；结果为两组均 strict 1/4。
2. 用修复后的 loss、完全相同 episode/seed 做 5k--10k 短重训；visual 与 torque 都从同一 base 初始化。
3. 固定未见 seed 做 8+8，再做至少 32+32，报告 paired difference、aligned、strict、min-xy 与 time-to-success。
4. 只有修复后仍有明确覆盖缺口才补采，按失败 sector/contact torque/初始 xy 分层。
5. 最终触觉结论必须加入同一 torque checkpoint 的 zero/shuffle 因果消融。
