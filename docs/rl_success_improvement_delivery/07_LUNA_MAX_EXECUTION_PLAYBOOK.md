# GPT-5.6 Luna Max 分阶段执行手册

## 1. 使用方式

不要一次要求模型实现全部阶段。每次只下发一个阶段，并要求模型：

1. 先阅读本目录 `README.md`、`00_PROJECT_CONTROL_AND_PREREQUISITES.md` 和当前阶段文档；
2. 检查工作树并保护既有改动；
3. 先实现最小可测试版本；
4. 运行与风险相称的测试；
5. 输出 `REPORT.md`、`METRICS.json` 和 `RUN_MANIFEST.json`；
6. 对照门禁自动给出 `PASS` 或 `FAIL`；
7. `FAIL` 时停止，不自动进入下一阶段。

## 2. 通用提示词头

每个阶段都使用以下提示词头：

```text
你正在 /root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla 工作。
先阅读 docs/rl_success_improvement_delivery/README.md、
00_PROJECT_CONTROL_AND_PREREQUISITES.md 和本阶段文档全文。

当前工作树包含用户已有修改。禁止 reset、checkout、clean，禁止覆盖历史实验结果。
所有新增 RL 代码放在 experiment/rl/，结果放在
experiment_results/rl_success_improvement_v1/<phase>/。

Actor 不得接收任何 peg/hole 真值、success、oracle 或未来力矩。
每完成一个子步骤先测试并记录证据；门禁失败就停止并报告。
不要把未实际执行的命令写成已完成结果。
```

## 3. Phase 0 提示词

```text
执行 Phase 0，只做基线冻结和快照 manifest，不训练模型。
阅读 01_PHASE_0_BASELINE_AND_MANIFEST.md，完成模型/配置/数据 SHA-256，
实现或复用同进程快照采集，建立不重叠的 train/validation/test manifest。
先完成 smoke 规模审计；若无法完成正式数量，报告实际完成单元和缺口，不伪造计数。
验证 split 泄漏、首帧一致性和 restore error，并生成 phase_0 交付物。
```

## 4. Phase 1 提示词

```text
执行 Phase 1，阅读 02_PHASE_1_REWARD_AND_ENVIRONMENT.md。
实现奖励函数、snapshot RL wrapper 和 action safety shield。
先写 reward、终止、安全层、Actor 观测隔离单元测试，再实现代码。
使用 Phase 0 train smoke 快照运行不超过 16 个 episode，
检查成功 return 高于失败、安全事件为负、所有 reward components 守恒。
不训练 SAC。生成 phase_1 报告并按文档门禁判定。
```

## 5. Phase 2 提示词

```text
执行 Phase 2，阅读 03_PHASE_2_RESIDUAL_SAC.md。
实现 causal torque GRU、Residual Actor、非对称双 Critic、replay buffer 和 SAC loss。
基础 SmolVLA 必须完全冻结。
先证明零 residual 等价、base 无梯度、Actor 无特权输入、zero/shuffle 同时作用于两层策略。
只进行 plumbing smoke，不进行正式训练；输出测试证据和 phase_2 门禁结论。
```

## 6. Phase 3 提示词

```text
执行 Phase 3，阅读 04_PHASE_3_TRAINING_AND_GATES.md。
严格按 Gate A→B→C→D→E 顺序执行。
test manifest 不得加载。
先进行 5k oracle residual BC，再做 5k SAC smoke，然后进行奖励消融。
只有每个门禁通过后才扩大预算。
checkpoint 选择使用安全优先的字典序，不使用累计 reward 或训练 loss 直接选模型。
记录所有实际命令、GPU、wall-clock、环境步、checkpoint 和 validation 指标。
```

## 7. Phase 4 提示词

```text
执行 Phase 4，阅读 05_PHASE_4_FORMAL_EVALUATION.md。
确认代码、模型、超参数和 test manifest SHA-256 已冻结。
先运行 16-pair identity smoke，再运行完整 384-pair test。
所有方法使用同一快照、flow-noise、推理采样数和安全层定义。
对 E4 运行 original/zero/shuffle，且干预同时作用于 base 和 residual。
生成整体、分层、配对 bootstrap、McNemar、安全和时间指标；
分别判定工程标准和论文级时序因果标准，不混为一个结论。
```

## 8. Phase 5 提示词

```text
仅当 Phase 4 工程标准 PASS 时执行 Phase 5。
阅读 06_PHASE_5_DISTILLATION_AND_RELEASE.md。
从非 test 轨迹构建蒸馏数据，使用逐样本 flow-matching loss 实现优势加权训练，
同时保留 unweighted 数据对照和 reference anchor。
完成回归验证后再运行冻结 test；如果蒸馏比 residual 系统低超过 3 个百分点，
保留 residual 系统作为最终部署方案，并在模型卡中如实说明。
```

## 9. 每阶段结束格式

要求模型最终返回：

```text
阶段：Phase N
判定：PASS / FAIL
完成内容：...
修改文件：...
测试：命令、退出码、关键结果
指标：...
未解决风险：...
下一阶段是否获准：是 / 否
报告路径：...
```

如果缺少 GPU、checkpoint、数据、存储空间或权限，应明确报告阻塞，不得用随机数据或伪造结果替代真实实验。
