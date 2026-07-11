#!/usr/bin/env python3
"""Convert raw peg-insert HDF5 demonstrations to a LeRobot v3 dataset.

Raw peg-insert contract:
  - state: [49]
  - actions: [7] = relative IK delta pose [6] + binary gripper [1]
  - rgb_table/rgb_wrist: uint8 [H,W,3]
  - gripper_torque: [1], mean of the two Franka finger applied torques

LeRobot contract:
  - observation.state: [49] by default, or compact [21] for official SmolVLA base compatibility
  - action: [7]
  - observation.images.camera1: [3,224,224] from rgb_table
  - observation.images.camera2: [3,224,224] from rgb_wrist
  - observation.gripper_torque: [30,1] causal history, newest at index -1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


FULL_STATE_DIM = 49
COMPACT_STATE_DIM = 21
ACTION_DIM = 7


def _select_state(state: np.ndarray, mode: str) -> np.ndarray:
    if mode == "full49":
        return state
    if mode == "compact21":
        # [joint_pos_rel(9), eef_pos(3), peg_pos(3), hole_pos(3), peg_to_hole_pos(3)]
        return np.concatenate([state[0:9], state[25:28], state[32:35], state[39:42], state[46:49]], axis=0)
    raise ValueError(f"Unsupported state mode: {mode}")


def _resize_rgb(img: np.ndarray, size: int = 224) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {img.shape}")
    if img.shape[:2] == (size, size):
        return img
    return np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def _causal_window(values: np.ndarray, t: int, size: int) -> np.ndarray:
    start = max(0, t - size + 1)
    history = values[start : t + 1]
    if history.shape[0] < size:
        history = np.concatenate([np.repeat(history[:1], size - history.shape[0], axis=0), history], axis=0)
    return history.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw peg-insert HDF5 demos to LeRobot dataset format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--task", type=str, default="Insert the peg into the hole")
    parser.add_argument("--robot-type", type=str, default="isaaclab_peg_insert_franka")
    parser.add_argument(
        "--state-mode",
        choices=("full49", "compact21"),
        default="full49",
        help=(
            "full49 preserves the Isaac policy observation. compact21 keeps proprioception and task geometry "
            "within official SmolVLA base max_state_dim=32: joint_pos(9), eef_pos(3), peg_pos(3), "
            "hole_pos(3), peg_to_hole_pos(3)."
        ),
    )
    parser.add_argument("--torque-window-size", type=int, default=30)
    parser.add_argument(
        "--torque-control",
        choices=("original", "zero", "shuffle_episode", "shuffle_global"),
        default="original",
        help=(
            "Torque negative-control transform. original preserves the measured torque. "
            "zero replaces torque with zeros. shuffle_episode permutes the raw torque sequence within each episode "
            "with a fixed seed before causal windowing, preserving marginal scale but breaking time alignment. "
            "shuffle_global permutes all raw torque samples across the dataset, which is useful when each episode "
            "has a nearly constant diagnostic torque mode."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=1000)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument(
        "--drop-terminal-frame",
        action="store_true",
        help="Drop the final frame from each demo before adding it to LeRobot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_dir / args.repo_id
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing LeRobot dataset: {root}")
    state_dim = FULL_STATE_DIM if args.state_mode == "full49" else COMPACT_STATE_DIM

    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        },
        "observation.gripper_torque": {
            "dtype": "float32",
            "shape": (args.torque_window_size, 1),
            "names": None,
        },
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        features=features,
        robot_type=args.robot_type,
        use_videos=args.use_videos,
    )

    with h5py.File(args.input, "r") as handle:
        if "data" not in handle:
            raise KeyError("Expected top-level 'data' group in HDF5.")
        recorded_fps = int(handle.attrs.get("fps", args.fps))
        if recorded_fps != args.fps:
            raise ValueError(f"FPS mismatch: HDF5={recorded_fps}, requested dataset FPS={args.fps}")
        demos = sorted(handle["data"].keys())
        if not demos:
            raise ValueError("HDF5 contains no demonstrations.")

        global_torque_pool = None
        global_torque_cursor = 0
        if args.torque_control == "shuffle_global":
            all_torques = []
            for demo_name in demos:
                demo = handle["data"][demo_name]
                if "gripper_torque" not in demo:
                    raise KeyError(f"{demo_name} is missing required field 'gripper_torque'")
                all_torques.append(np.asarray(demo["gripper_torque"], dtype=np.float32).reshape(-1, 1))
            global_torque_pool = np.concatenate(all_torques, axis=0)
            rng = np.random.default_rng(args.shuffle_seed)
            global_torque_pool = global_torque_pool[rng.permutation(global_torque_pool.shape[0])]

        print(f"Converting {len(demos)} demos from {args.input}")
        for demo_index, demo_name in enumerate(demos, 1):
            demo = handle["data"][demo_name]
            for key in ("state", "actions", "rgb_table", "rgb_wrist", "gripper_torque"):
                if key not in demo:
                    raise KeyError(f"{demo_name} is missing required field {key!r}")

            actions = np.asarray(demo["actions"], dtype=np.float32)
            states = np.asarray(demo["state"], dtype=np.float32)
            torques = np.asarray(demo["gripper_torque"], dtype=np.float32).reshape(-1, 1)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(f"{demo_name}/actions shape {actions.shape}, expected [T,{ACTION_DIM}]")
            if states.ndim != 2 or states.shape[1] != FULL_STATE_DIM:
                raise ValueError(f"{demo_name}/state shape {states.shape}, expected [T,{FULL_STATE_DIM}]")
            if torques.shape != (actions.shape[0], 1):
                raise ValueError(f"{demo_name}/gripper_torque shape {torques.shape}, expected [T,1]")
            if not np.isfinite(actions).all() or not np.isfinite(states).all() or not np.isfinite(torques).all():
                raise ValueError(f"{demo_name} contains NaN or Inf in state/action/torque")
            if args.torque_control == "zero":
                torques = np.zeros_like(torques)
            elif args.torque_control == "shuffle_episode":
                rng = np.random.default_rng(args.shuffle_seed + demo_index)
                torques = torques[rng.permutation(torques.shape[0])]
            elif args.torque_control == "shuffle_global":
                assert global_torque_pool is not None
                next_cursor = global_torque_cursor + torques.shape[0]
                if next_cursor > global_torque_pool.shape[0]:
                    raise RuntimeError("Global torque shuffle pool exhausted unexpectedly")
                torques = global_torque_pool[global_torque_cursor:next_cursor]
                global_torque_cursor = next_cursor

            t_raw = actions.shape[0]
            t_used = t_raw - 1 if args.drop_terminal_frame else t_raw
            if t_used <= 0:
                raise ValueError(f"{demo_name} has no usable frames")
            for key in ("rgb_table", "rgb_wrist"):
                if len(demo[key]) != t_raw:
                    raise ValueError(f"{demo_name}/{key} length {len(demo[key])}, expected {t_raw}")

            suffix = " (terminal frame dropped)" if args.drop_terminal_frame else ""
            print(f"  [{demo_index}/{len(demos)}] {demo_name}: {t_used} frames{suffix}")
            for t in range(t_used):
                ds.add_frame(
                    {
                        "task": args.task,
                        "observation.state": _select_state(states[t], args.state_mode),
                        "action": actions[t],
                        "observation.images.camera1": _resize_rgb(demo["rgb_table"][t]),
                        "observation.images.camera2": _resize_rgb(demo["rgb_wrist"][t]),
                        "observation.gripper_torque": _causal_window(torques, t, args.torque_window_size),
                    }
                )
            ds.save_episode()

    ds.finalize()
    print(f"Successfully wrote LeRobot dataset to: {ds.root}")
    print(f"Interface: state=[{state_dim}], action=[{ACTION_DIM}], camera1/2=[3,224,224], torque=[{args.torque_window_size},1]")
    print(f"State mode: {args.state_mode}")
    print(f"Torque control: {args.torque_control}")


if __name__ == "__main__":
    main()
