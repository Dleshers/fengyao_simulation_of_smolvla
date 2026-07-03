# Remote Codex Setup: SmolVLA Tactile + Torque Window Simulation (Legacy Linear Version)

> **Deprecated for the current main experiment.** This document and the
> `SETUP_torque_window_*` patches use `flatten -> Linear`, not LSTM. For the
> visual-vs-torque-LSTM experiment, read `LSTM_ACTION_EXPERT_EXPERIMENT.md`
> and use the updated SmolVLA source files.

This document is written for a Codex instance running on the remote GPU machine.
It summarizes the project background, the intended experiment plan, and the setup commands needed to reproduce the simulation pipeline.

## 1. Project Background

The project studies whether adding force / tactile signals to SmolVLA improves or destabilizes robot behavior in a full IsaacLab simulation rollout.

The current target is not only offline training loss. The required validation is:

```text
Load a SmolVLA policy in full IsaacLab simulation
Receive visual/state/tactile/torque observations online
Send actions back to the simulated robot
Observe whether the robot moves stably or behaves erratically
```

The original tactile pipeline uses two repositories:

```text
IsaacLab-Tactile
  - Isaac Sim / IsaacLab simulation
  - expert demo collection
  - eval_server.py remote simulation backend

lerobot-tactile
  - HDF5 -> LeRobot dataset conversion
  - SmolVLA training
  - lerobot-eval client connecting to eval_server.py
```

The base tactile patch fixes the full chain:

```text
collection data fields
  == conversion data fields
  == online eval observation fields
```

This equality is the most important rule. If training and eval use different tactile or torque representations, the policy can appear to "randomly move" even if the model itself is not the problem.

## 2. Current Plan

There are now three experiment modes.

### A. Baseline Tactile SmolVLA

Uses the original tactile grid:

```text
observation.tactile.force_grid: [2, 10, 12, 3]
```

Use this to verify the original visual+tactile pipeline.

### B. Gripper Torque Window Variant

Adds a causal gripper torque history:

```text
observation.gripper_torque: [torque_window_size, 1]
default torque_window_size = 32
```

In SmolVLA, this is injected directly into the action expert suffix stream:

```text
gripper torque window
  -> Linear(window * 1, expert_hidden)
  -> torque token
  -> action expert suffix
```

### C. Whole-Body Torque Window Variant

Adds a causal whole-body joint torque history:

```text
observation.joint_torque: [torque_window_size, joint_torque_dim]
default torque_window_size = 32
default joint_torque_dim = 9
```

The expected 9D layout is:

```text
7 arm joint torques + 2 gripper joint torques
```

It also injects directly into the action expert suffix stream:

```text
joint torque window
  -> Linear(window * dim, expert_hidden)
  -> torque token
  -> action expert suffix
```

Do not apply the gripper torque and whole-body torque variants at the same time. Pick one variant per clean clone.

## 3. Required Remote Machine

Recommended environment:

```text
Linux remote GPU machine
NVIDIA GPU + working driver
Isaac Sim 4.5 standalone
Python 3.10 for IsaacLab / TacEx
Conda or micromamba
Git
```

TacEx currently requires:

```text
Isaac Sim 4.5
Python 3.10
--enable_cameras for tactile rendering
```

Isaac Sim 4.5 usually requires a manual NVIDIA download. Place it at:

```text
~/.local/share/ov/pkg/isaac-sim-4.5.0
```

## 4. Workspace Layout

Recommended remote layout:

```text
~/smolvla_tactile_sim/
  IsaacLab-Tactile/
  lerobot-tactile/
  TacEx/
  patches/
    SETUP_minimal_pickplace_IsaacLab.patch
    SETUP_minimal_pickplace_lerobot.patch
    SETUP_torque_window_gripper_IsaacLab.patch
    SETUP_torque_window_gripper_lerobot.patch
    SETUP_torque_window_fullbody_IsaacLab.patch
    SETUP_torque_window_fullbody_lerobot.patch
    SETUP_TORQUE_WINDOW_PATCHES.md
    REMOTE_CODEX_SETUP_TORQUE_WINDOW_SIM.md
```

Copy the `simulation/*.patch` files from the local project into `~/smolvla_tactile_sim/patches/` on the remote machine.

## 5. Clone Source Repositories

```bash
mkdir -p ~/smolvla_tactile_sim
cd ~/smolvla_tactile_sim

git clone --recurse-submodules https://github.com/rechim25/IsaacLab-Tactile.git
git clone https://github.com/rechim25/lerobot-tactile.git
git clone --recurse-submodules https://github.com/DH-Ng/TacEx.git
```

Before applying patches, keep the clones clean:

```bash
cd ~/smolvla_tactile_sim/IsaacLab-Tactile
git status --short

cd ~/smolvla_tactile_sim/lerobot-tactile
git status --short
```

## 6. Choose and Apply Patches

Choose exactly one mode.

### Option A: Baseline Tactile

```bash
cd ~/smolvla_tactile_sim/IsaacLab-Tactile
git apply --check ../patches/SETUP_minimal_pickplace_IsaacLab.patch
git apply ../patches/SETUP_minimal_pickplace_IsaacLab.patch

cd ~/smolvla_tactile_sim/lerobot-tactile
git apply --check ../patches/SETUP_minimal_pickplace_lerobot.patch
git apply ../patches/SETUP_minimal_pickplace_lerobot.patch
```

### Option B: Gripper Torque Window

```bash
cd ~/smolvla_tactile_sim/IsaacLab-Tactile
git apply --check ../patches/SETUP_torque_window_gripper_IsaacLab.patch
git apply ../patches/SETUP_torque_window_gripper_IsaacLab.patch

cd ~/smolvla_tactile_sim/lerobot-tactile
git apply --check ../patches/SETUP_torque_window_gripper_lerobot.patch
git apply ../patches/SETUP_torque_window_gripper_lerobot.patch
```

### Option C: Whole-Body Torque Window

```bash
cd ~/smolvla_tactile_sim/IsaacLab-Tactile
git apply --check ../patches/SETUP_torque_window_fullbody_IsaacLab.patch
git apply ../patches/SETUP_torque_window_fullbody_IsaacLab.patch

cd ~/smolvla_tactile_sim/lerobot-tactile
git apply --check ../patches/SETUP_torque_window_fullbody_lerobot.patch
git apply ../patches/SETUP_torque_window_fullbody_lerobot.patch
```

If `git apply --check` fails, stop and inspect the target repository version. Do not force-apply.

## 7. Install IsaacLab + TacEx

Install Isaac Sim 4.5 standalone first. Expected path:

```bash
mkdir -p ~/.local/share/ov/pkg/isaac-sim-4.5.0
# unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip into that directory
```

Then install IsaacLab-Tactile:

```bash
cd ~/smolvla_tactile_sim/IsaacLab-Tactile
ln -s ~/.local/share/ov/pkg/isaac-sim-4.5.0 _isaac_sim

./isaaclab.sh --conda
conda activate env_isaaclab
python --version  # must be 3.10.x

./isaaclab.sh --install
```

Install TacEx:

```bash
cd ~/smolvla_tactile_sim/TacEx
conda activate env_isaaclab
./tacex.sh --install
```

Smoke test:

```bash
cd ~/smolvla_tactile_sim/IsaacLab-Tactile
conda activate env_isaaclab

./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0 \
  --num_envs 1 \
  --enable_cameras
```

If CUDA error 999 appears, kill stale Isaac Sim processes or reboot the remote machine.

## 8. Install LeRobot / SmolVLA Environment

Use a separate Python 3.10 environment:

```bash
cd ~/smolvla_tactile_sim/lerobot-tactile

conda create -n smolvla python=3.10 -y
conda activate smolvla

pip install -e ".[smolvla]"
```

If installation needs network access, run from a shell with internet access or configure the university proxy / package mirror.

## 9. Collect Expert Demos

Run from IsaacLab-Tactile:

```bash
cd ~/smolvla_tactile_sim/IsaacLab-Tactile
conda activate env_isaaclab

./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
  --num_envs 4 \
  --num_demos 100 \
  --headless \
  --enable_cameras \
  --rendering_mode quality \
  --save_demos \
  --output_dir ./datasets/pick_place_basket_tacex
```

Expected output:

```text
~/smolvla_tactile_sim/IsaacLab-Tactile/datasets/pick_place_basket_tacex/data.hdf5
```

For the torque variants, the patched recorder adds:

```text
gripper variant:
  gripper_torque: [T, 1]

whole-body variant:
  joint_torque: [T, 9]
```

## 10. Convert Dataset

Run from lerobot-tactile.

### Baseline Tactile

```bash
cd ~/smolvla_tactile_sim/lerobot-tactile
conda activate smolvla

python convert_pick_place_basket_tacex.py \
  --input ../IsaacLab-Tactile/datasets/pick_place_basket_tacex/data.hdf5 \
  --output-dir ./datasets \
  --repo-id pick_place_basket_tactile \
  --tactile-source force_grid_geometric
```

### Gripper Torque Window

```bash
python convert_pick_place_basket_tacex.py \
  --input ../IsaacLab-Tactile/datasets/pick_place_basket_tacex/data.hdf5 \
  --output-dir ./datasets \
  --repo-id pick_place_basket_gripper_torque_window \
  --tactile-source force_grid_geometric \
  --torque-window-size 32
```

### Whole-Body Torque Window

```bash
python convert_pick_place_basket_tacex.py \
  --input ../IsaacLab-Tactile/datasets/pick_place_basket_tacex/data.hdf5 \
  --output-dir ./datasets \
  --repo-id pick_place_basket_fullbody_torque_window \
  --tactile-source force_grid_geometric \
  --torque-window-size 32 \
  --joint-torque-dim 9
```

## 11. Train SmolVLA

### Baseline Tactile

```bash
lerobot-train \
  --dataset.repo_id=pick_place_basket_tactile \
  --dataset.root=./datasets/pick_place_basket_tactile \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.push_to_hub=false \
  --policy.use_tactile=true \
  --policy.num_fingertips=2 \
  --policy.use_arm_hand_feature_enhancement=true \
  --policy.arm_indices='[0,1,2,3,4,5]' \
  --policy.hand_indices='[6]' \
  --policy.aux_loss_lambda=1.0 \
  --policy.empty_cameras=1 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/smolvla_tactile_run
```

### Gripper Torque Window

```bash
lerobot-train \
  --dataset.repo_id=pick_place_basket_gripper_torque_window \
  --dataset.root=./datasets/pick_place_basket_gripper_torque_window \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.push_to_hub=false \
  --policy.use_tactile=true \
  --policy.use_torque_window=true \
  --policy.torque_window_key=observation.gripper_torque \
  --policy.torque_window_size=32 \
  --policy.torque_window_dim=1 \
  --policy.empty_cameras=1 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/smolvla_gripper_torque_window_run
```

### Whole-Body Torque Window

```bash
lerobot-train \
  --dataset.repo_id=pick_place_basket_fullbody_torque_window \
  --dataset.root=./datasets/pick_place_basket_fullbody_torque_window \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.push_to_hub=false \
  --policy.use_tactile=true \
  --policy.use_torque_window=true \
  --policy.torque_window_key=observation.joint_torque \
  --policy.torque_window_size=32 \
  --policy.torque_window_dim=9 \
  --policy.empty_cameras=1 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/smolvla_fullbody_torque_window_run
```

For first debugging, reduce `--steps` and `--num_demos` to validate the whole loop quickly.

## 12. Full Simulation Evaluation

Evaluation always uses two processes on the remote GPU machine.

### Terminal A: IsaacLab eval server

Baseline tactile:

```bash
cd ~/smolvla_tactile_sim/IsaacLab-Tactile
conda activate env_isaaclab

python scripts/eval_server.py --host '*' --port 5555 \
  --env Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0 \
  --tactile-grid-source height_map
```

Gripper torque window:

```bash
python scripts/eval_server.py --host '*' --port 5555 \
  --env Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0 \
  --tactile-grid-source height_map \
  --send-gripper-torque-window \
  --torque-window-size 32
```

Whole-body torque window:

```bash
python scripts/eval_server.py --host '*' --port 5555 \
  --env Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0 \
  --tactile-grid-source height_map \
  --send-joint-torque-window \
  --torque-window-size 32 \
  --joint-torque-dim 9
```

### Terminal B: LeRobot eval client

Baseline tactile:

```bash
cd ~/smolvla_tactile_sim/lerobot-tactile
conda activate smolvla

lerobot-eval \
  --policy.path=outputs/smolvla_tactile_run/checkpoints/last/pretrained_model \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost \
  --env.server_port=5555 \
  --env.task=pick_place \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.rgb_table":"observation.images.camera1"}'
```

Gripper torque window:

```bash
lerobot-eval \
  --policy.path=outputs/smolvla_gripper_torque_window_run/checkpoints/last/pretrained_model \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost \
  --env.server_port=5555 \
  --env.task=pick_place \
  --env.include_gripper_torque_window=true \
  --env.torque_window_size=32 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.rgb_table":"observation.images.camera1"}'
```

Whole-body torque window:

```bash
lerobot-eval \
  --policy.path=outputs/smolvla_fullbody_torque_window_run/checkpoints/last/pretrained_model \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost \
  --env.server_port=5555 \
  --env.task=pick_place \
  --env.include_joint_torque_window=true \
  --env.torque_window_size=32 \
  --env.joint_torque_dim=9 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.rgb_table":"observation.images.camera1"}'
```

Use `--eval.batch_size=1` for visual inspection of whether the robot is moving erratically.

## 13. Debugging Priorities

If the robot moves randomly after enabling tactile or torque, check these first:

```text
1. Was the same patch variant used for collection, conversion, training, and eval?
2. Is the eval server sending the required torque window flag?
3. Does the dataset contain the expected feature?
4. Do feature shapes match?
5. Are camera keys mapped correctly through rename_map?
6. Is action space the same between training and eval?
7. Is the checkpoint config.json saving "type": "smolvla"?
8. Is tactile source aligned?
```

For tactile source alignment:

```text
training conversion:
  --tactile-source force_grid_geometric

online eval:
  --tactile-grid-source height_map
```

This is intentional because the geometric force grid is generated from the height map.

## 14. Expected Deliverables

The remote Codex should produce:

```text
1. A clean patched IsaacLab-Tactile clone
2. A clean patched lerobot-tactile clone
3. A small collected demo dataset
4. A converted LeRobot dataset
5. At least one short training run
6. A full simulation eval rollout video / observation
7. A note on whether tactile or torque conditioning causes unstable motion
```

For the first pass, prefer a small smoke test:

```text
num_demos: 5-10
steps: 100-500
n_episodes: 1-3
batch_size: 1
```

Only scale to larger runs after the full online loop works.
