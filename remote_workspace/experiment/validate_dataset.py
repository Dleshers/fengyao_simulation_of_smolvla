#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    sample = dataset[0]
    required_shapes = {
        "observation.state": (9,),
        "observation.images.camera1": (3, 224, 224),
        "observation.images.camera2": (3, 224, 224),
        "action": (8,),
    }
    for feature, expected in required_shapes.items():
        if feature not in sample:
            raise KeyError(f"Dataset is missing {feature}")
        actual = tuple(torch.as_tensor(sample[feature]).shape[-len(expected) :])
        if actual != expected:
            raise ValueError(f"Expected {feature} [...,{expected}], got {actual}")

    key = "observation.gripper_torque"
    if key not in sample:
        raise KeyError(f"Dataset is missing {key}")
    torque = torch.as_tensor(sample[key])
    if tuple(torque.shape[-2:]) != (args.window_size, 1):
        raise ValueError(f"Expected {key} [...,{args.window_size},1], got {tuple(torque.shape)}")
    if not torch.isfinite(torque).all():
        raise ValueError(f"{key} contains NaN or Inf")

    indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=int)
    states = torch.stack([torch.as_tensor(dataset[int(i)]["observation.state"]).float() for i in indices])
    actions = torch.stack([torch.as_tensor(dataset[int(i)]["action"]).float() for i in indices])
    torques = torch.stack([torch.as_tensor(dataset[int(i)][key]).float() for i in indices])
    for name, values in (("state", states), ("action", actions), ("torque", torques)):
        if not torch.isfinite(values).all():
            raise ValueError(f"Sampled {name} contains NaN or Inf")

    print("OK: shared baseline/tactile schema is state[9], camera1/2[3,224,224], action[8]")
    print(f"OK: {key} dtype={torque.dtype} shape={tuple(torque.shape)}; index -1 must be newest")
    print(f"state range: [{states.min().item():.6g}, {states.max().item():.6g}]")
    print(f"action range: [{actions.min().item():.6g}, {actions.max().item():.6g}]")
    print(f"torque range: [{torques.min().item():.6g}, {torques.max().item():.6g}]")


if __name__ == "__main__":
    main()
