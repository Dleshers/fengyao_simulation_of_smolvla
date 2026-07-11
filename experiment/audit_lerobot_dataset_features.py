#!/usr/bin/env python3
"""Small LeRobot dataset interface/statistics audit.

This is intentionally lightweight: it loads a local LeRobot dataset, checks the
core SmolVLA feature shapes, and reports basic torque-window statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local LeRobot feature shapes and torque statistics.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-dim", type=int, default=49)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = LeRobotDataset(args.repo_id, root=args.root)
    n = ds.num_frames if args.max_frames <= 0 else min(ds.num_frames, args.max_frames)
    if n <= 0:
        raise ValueError("Dataset has no frames")

    first = ds[0]
    expected = {
        "observation.state": (args.state_dim,),
        "action": (args.action_dim,),
        "observation.images.camera1": (3, 224, 224),
        "observation.images.camera2": (3, 224, 224),
        "observation.gripper_torque": (30, 1),
    }
    for key, shape in expected.items():
        actual = tuple(first[key].shape)
        if actual != shape:
            raise AssertionError(f"{key}: expected {shape}, got {actual}")

    torques = torch.stack([ds[i]["observation.gripper_torque"].float() for i in range(n)])
    newest = torques[:, -1, 0]
    print(f"repo_id={args.repo_id}")
    print(f"root={args.root}")
    print(f"episodes={ds.num_episodes} frames={ds.num_frames} audited_frames={n}")
    for key, shape in expected.items():
        print(f"OK {key} shape={shape} dtype={first[key].dtype}")
    print(
        "torque_newest "
        f"min={float(newest.min()):.6g} max={float(newest.max()):.6g} "
        f"mean={float(newest.mean()):.6g} std={float(newest.std(unbiased=False)):.6g}"
    )
    print(
        "torque_window "
        f"min={float(torques.min()):.6g} max={float(torques.max()):.6g} "
        f"mean={float(torques.mean()):.6g} std={float(torques.std(unbiased=False)):.6g}"
    )


if __name__ == "__main__":
    main()
