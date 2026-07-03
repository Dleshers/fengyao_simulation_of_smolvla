# Torque Window Patch Variants

These patches are derived from the minimal pick-and-place tactile patches.
The original two files are kept unchanged.

## Gripper torque window

Use these together:

```bash
git apply /path/to/SETUP_torque_window_gripper_IsaacLab.patch
git apply /path/to/SETUP_torque_window_gripper_lerobot.patch
```

Dataset / policy contract:

```text
observation.gripper_torque: [torque_window_size, 1]
```

IsaacLab eval server:

```bash
python scripts/eval_server.py --host '*' --port 5555 \
  --env Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0 \
  --send-gripper-torque-window \
  --torque-window-size 32
```

LeRobot conversion:

```bash
python convert_pick_place_basket_tacex.py \
  --input ../IsaacLab-Tactile/datasets/pick_place_basket_tacex/data.hdf5 \
  --output-dir ./datasets \
  --repo-id pick_place_basket_gripper_torque_window \
  --tactile-source force_grid_geometric \
  --torque-window-size 32
```

## Whole-body torque window

Use these together:

```bash
git apply /path/to/SETUP_torque_window_fullbody_IsaacLab.patch
git apply /path/to/SETUP_torque_window_fullbody_lerobot.patch
```

Dataset / policy contract:

```text
observation.joint_torque: [torque_window_size, joint_torque_dim]
default joint_torque_dim = 9
```

IsaacLab eval server:

```bash
python scripts/eval_server.py --host '*' --port 5555 \
  --env Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0 \
  --send-joint-torque-window \
  --torque-window-size 32 \
  --joint-torque-dim 9
```

LeRobot conversion:

```bash
python convert_pick_place_basket_tacex.py \
  --input ../IsaacLab-Tactile/datasets/pick_place_basket_tacex/data.hdf5 \
  --output-dir ./datasets \
  --repo-id pick_place_basket_fullbody_torque_window \
  --tactile-source force_grid_geometric \
  --torque-window-size 32 \
  --joint-torque-dim 9
```

## Model path

Both variants inject the torque window directly into the SmolVLA action expert suffix stream:

```text
torque window -> Linear(window * dim, expert_hidden) -> torque token -> action expert suffix
```

This intentionally bypasses the tactile grid encoder path. The tactile grid patches are still retained so the original visual+tactile evaluation pipeline remains available.
