# 触觉仿真 Pipeline 搭建教程（pick-and-place）

目标：在 rechim25 原始仓库基础上，打两个补丁，跑通 **pick-and-place（basket 抓放）** 的整条触觉 pipeline：**采集 → 转换 → 训练 → 评估**。
论文里的 peg 插入、tilt/recovery、各种触觉融合消融都不在本文范围。

> 本文命令已对着代码逐条核对过（独立复核）。224 分辨率、180s 超时等都是默认值，不用手动设。

---

## 0. 整条链路

```
[IsaacLab-Tactile] (Isaac Sim 4.5)            [lerobot-tactile] (普通 PyTorch)
 采集 sm 脚本 → data.hdf5 → 转换 → LeRobot 数据集 → lerobot-train → checkpoint
                                                                       │
        eval_server.py (仿真后端) ◄── socket ──► lerobot-eval (客户端) ┘
```
- **采集 + 评估后端**跑在 IsaacLab（需 Isaac Sim 4.5 + GPU）；**训练 + 评估客户端**跑在 lerobot。
- 评估时两端同时开：IsaacLab 起 `eval_server.py`，lerobot 起 `lerobot-eval` 连过去。

---

## 1. 拿代码：clone 原版 + 打两个补丁

两个仓库都是 rechim25 原版，我们只在上面加了"跑通 pick-and-place 必需"的少量改动（IsaacLab 3 文件 / lerobot 11 文件），打包成两个 patch。

```bash
# 仿真/采集仓库
git clone --recurse-submodules https://github.com/rechim25/IsaacLab-Tactile.git
cd IsaacLab-Tactile
git apply /path/to/SETUP_minimal_pickplace_IsaacLab.patch
cd ..

# 训练/评估仓库
git clone https://github.com/rechim25/lerobot-tactile.git
cd lerobot-tactile
git apply /path/to/SETUP_minimal_pickplace_lerobot.patch
cd ..
```
> 两个 patch 已实测能干净打到原版 `main`。打完 `git diff --stat` 可看到改了哪些文件。

---

## 2. 环境安装（照原仓库文档，约束很硬）

**TacEx（触觉传感器仿真）只支持 Isaac Sim 4.5 + Python 3.10**（5.x 是 3.11，装不上）。详细步骤见 IsaacLab-Tactile 仓库的 `INSTALL_WITH_TACEX.md`，要点：

```bash
# 1) 装 Isaac Sim 4.5 standalone（zip 解压是扁平的，先建目录再解）
mkdir -p ~/.local/share/ov/pkg/isaac-sim-4.5.0
unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip -d ~/.local/share/ov/pkg/isaac-sim-4.5.0

# 2) IsaacLab 软链到 Isaac Sim + 建 conda 环境（必须 3.10）+ 装
cd IsaacLab-Tactile
ln -s ~/.local/share/ov/pkg/isaac-sim-4.5.0 _isaac_sim
./isaaclab.sh --conda && conda activate env_isaaclab
python --version          # 必须 3.10.x
./isaaclab.sh --install

# 3) 装 TacEx（带 submodule）
git clone --recurse-submodules https://github.com/DH-Ng/TacEx
cd TacEx && ./tacex.sh --install && cd ..

# 4) 自检（务必带 --enable_cameras，触觉靠渲染算）
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0 --num_envs 1 --enable_cameras

# 5) lerobot 训练环境（另一个 conda，不需要 Isaac Sim）
cd ../lerobot-tactile
conda create -n smolvla python=3.10 -y && conda activate smolvla
pip install -e ".[smolvla]"
```
> 坑：所有 TacEx 脚本都要 `--enable_cameras`；`CUDA error 999` → 关掉残留 Isaac Sim 进程，不行就重启。

---

## 3. 按顺序跑起来

### ① 采集（IsaacLab，`env_isaaclab` 环境）
```bash
cd IsaacLab-Tactile && conda activate env_isaaclab

./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
  --num_envs 4 --num_demos 100 --headless --enable_cameras \
  --rendering_mode quality --save_demos \
  --output_dir ./datasets/pick_place_basket_tacex
# ⚠️ 是 --output_dir（目录），产出文件 = ./datasets/pick_place_basket_tacex/data.hdf5
```

### ② 检查数据（可选）
```bash
python scripts/tools/inspect_hdf5.py ./datasets/pick_place_basket_tacex/data.hdf5 \
  --samples 1 --plot --trajectory --forces
python scripts/tools/inspect_hdf5.py ./datasets/pick_place_basket_tacex/data.hdf5 \
  --video --demo-idx 0 --video-output demo.mp4 --fps 30
```

### ③ 转换 HDF5 → LeRobot 数据集（lerobot，`smolvla` 环境）
```bash
cd ../lerobot-tactile && conda activate smolvla

python convert_pick_place_basket_tacex.py \
  --input    ../IsaacLab-Tactile/datasets/pick_place_basket_tacex/data.hdf5 \
  --output-dir ./datasets \
  --repo-id  pick_place_basket_tacex_lerobot \
  --tactile-source force_grid_geometric        # 默认值；触觉表征=几何压陷力栅格
# ⚠️ 必须显式传 --output-dir（脚本默认值是原作者的硬编码路径，不传会写到不存在的目录）
```

### ④ 训练（触觉 vs 纯视觉只差一个 flag）
```bash
lerobot-train \
  --dataset.repo_id=pick_place_basket_tacex_lerobot \
  --dataset.root=./datasets/pick_place_basket_tacex_lerobot \
  --policy.type=smolvla --policy.device=cuda \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.push_to_hub=false \
  --policy.use_tactile=true \                  # ← 纯视觉对照改 false，其余不动
  --policy.num_fingertips=2 \
  --policy.use_arm_hand_feature_enhancement=true \
  --policy.arm_indices='[0,1,2,3,4,5]' --policy.hand_indices='[6]' \
  --policy.aux_loss_lambda=1.0 --policy.empty_cameras=1 \
  --batch_size=8 --steps=20000 \
  --output_dir=outputs/smolvla_tactile_run \
  --wandb.enable=true --wandb.project=smolvla-tactile
```
> 精度坑：触觉那条路对 bf16 敏感（会打崩触觉），想稳就用 fp32；要混合精度设环境变量 `ACCELERATE_MIXED_PRECISION=bf16`（`--policy.use_amp` 是空操作），且触觉/视觉两臂必须同精度。

### ⑤ 评估（两端同时开）
**A. 先在 IsaacLab 起仿真后端：**
```bash
cd IsaacLab-Tactile && conda activate env_isaaclab
python scripts/eval_server.py --host '*' --port 5555 \
  --env Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0
# ⚠️ --env 要传完整 gym id，不能写 pick_place_basket；--img_height/width 默认已是 224
```
**B. 再在 lerobot 起评估客户端：**
```bash
cd ../lerobot-tactile && conda activate smolvla
lerobot-eval \
  --policy.path=outputs/smolvla_tactile_run/checkpoints/last/pretrained_model \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost --env.server_port=5555 \
  --env.task=pick_place \
  --eval.n_episodes=10 --eval.batch_size=10 \
  --rename_map='{"observation.images.rgb_table":"observation.images.camera1"}'
# observation_height/width 默认 224、timeout_ms 默认 180000，已和后端对齐，无需手动设
# --rename_map 按你数据集的相机 key 实际填（把数据集相机名映射成策略期望名）
```

---

## 4. 想改哪里就动哪里（文件位置）

### ① 任务环境（IsaacLab）
目录：`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/<任务>/`（如 `pick_place_basket/`、`stack/`）
- `<任务>_env_cfg.py` + `config/franka/*_tacex_env_cfg.py` — 场景/机器人/物体/相机/动作空间
- `mdp/terminations.py` — **成功/失败判定**（换任务最该先检查的）
- `mdp/observations.py`（视觉/状态）、`mdp/tacex_observations.py`（触觉栅格）、`mdp/events.py`（随机化）
- `config/franka/__init__.py` — `gym.register(...)` 注册任务 id
- `scripts/environments/state_machine/<任务>_tacex_sm.py` — 脚本化专家采集器

### ② 模型架构（lerobot，`src/lerobot/policies/smolvla/`）
- `tactile.py` — 触觉模块：`TactileCAE`（触觉编码器）、`TactileEmbedding`（栅格→token）、`ArmHandFeatureEnhancement`
- `configuration_smolvla.py` — 所有 `--policy.*` 超参 + 一致性校验
- `modeling_smolvla.py` — 策略主体：触觉提取、token 拼接、和 VLM/action expert 的融合接线
- `smolvlm_with_expert.py` — VLM 主干 + action expert

### ③ 评估（两端）
- IsaacLab 后端：`scripts/eval_server.py` — 成功判定、观测打包、触觉栅格来源（`--tactile-grid-source`，可选 `height_map`/`photometric_rgb`，默认 `height_map`，对应几何触觉）
- lerobot 客户端：`src/lerobot/envs/isaaclab_tactile_remote.py`（远程 env、通信、观测转换）、`src/lerobot/envs/configs.py`（`IsaacLabTactileRemoteEnv` 配置类）、`src/lerobot/scripts/lerobot_eval.py`（rollout/成功率/渲染）、`src/lerobot/configs/eval.py`（`EvalPipelineConfig`，含 `rename_map`）

---

## 5. 这两个 patch 到底改了什么（为什么必需）

原版 clone 下来，**采集数据格式和评估链路是跑不对的**，patch 修的就是这些。

**IsaacLab patch（3 文件，前期环境修复）**
- `pick_place_basket_tacex_sm.py` — 写入 10×12×3 触觉力栅格（`force_grid_geometric_*` + `height_map_*`）+ HDF5 写入重试。这是触觉表征的"数据契约"。
- `convert_pick_place_basket_tacex.py` — 加 `--tactile-source force_grid_geometric`，让转换出的触觉与采集字段一致。
- `eval_server.py` — 加 `--tactile-grid-source`，让评估在线算的触觉与训练数据一致；并修了 eval 成功判定 + 渲染。

> 一致性铁律：**采集写入 = 转换读取 = 评估在线计算**，触觉表征必须三处一致，否则评估分数全错。

**lerobot patch（11 文件，评估链路修复 —— 原版 train→eval 交接的一堆坑）**
- `configs/policies.py` — 把 `type` 写进 `config.json`，否则评估**加载不出模型**
- `policies/factory.py` — 刷新失配的 input/output features + 应用 `rename_map`，否则训练起不来
- `configuration_smolvla.py` — 修触觉归一化退化（强制 MEAN_STD）
- `envs/configs.py` / `isaaclab_tactile_remote.py` — 远程环境默认对齐后端（224 分辨率 / 180s 超时 / episode 长度）+ 渲染取帧
- `envs/factory.py` — `autoreset_mode=SAME_STEP`（成功率统计正确的前提）
- `envs/utils.py` / `processor/env_processor.py` — 远程观测 key 处理 + 图像 uint8 HWC→float CHW
- `scripts/lerobot_eval.py` — 适配 Gymnasium 1.x 的成功率解析 + 渲染 done_mask（避免录到 reset 后的帧）
- `configs/eval.py` / `utils/train_utils.py` — eval 渲染字段 + checkpoint 存盘校验

---

## 附：换成 peg 插入等其它任务
不在本文范围。要点提示：peg 还需改 `peg_insert/mdp/terminations.py` 的成功判定（原版要求约 70mm 深度、物理不可能，会一条都存不下）和 `peg_insert_env_cfg.py` 的物体尺寸/抓取位姿。需要时另说。
