#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))

def _scalar_int(value, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"Expected scalar {name}, got shape {tuple(tensor.shape)}")
    return int(tensor.item())


def audit_causal_torque_windows(dataset, key: str, window_size: int, limit: int) -> None:
    checked = min(limit, len(dataset))
    if checked == 0:
        raise ValueError("Dataset is empty")

    previous_window = None
    previous_episode = None
    episode_starts = 0
    transitions = 0
    for index in range(checked):
        sample = dataset[index]
        if "episode_index" not in sample:
            raise KeyError("Dataset samples must expose episode_index for boundary auditing")
        episode = _scalar_int(sample["episode_index"], "episode_index")
        window = torch.as_tensor(sample[key]).float()
        if tuple(window.shape[-2:]) != (window_size, 1):
            raise ValueError(f"Sample {index} has invalid torque shape {tuple(window.shape)}")
        window = window.reshape(window_size, 1)

        if previous_episode != episode:
            episode_starts += 1
            if not torch.allclose(window, window[-1:].expand_as(window), rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"Episode {episode} starts with incorrect padding at dataset index {index}; "
                    f"all {window_size} values must repeat the first valid sample"
                )
        else:
            transitions += 1
            if not torch.allclose(window[:-1], previous_window[1:], rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"Torque window is not causal/contiguous at dataset index {index} "
                    f"inside episode {episode}"
                )

        previous_window = window
        previous_episode = episode

    print(
        f"OK: audited {checked} sequential torque windows "
        f"({episode_starts} episode starts, {transitions} within-episode transitions)"
    )


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--sequence-checks",
        type=int,
        default=512,
        help="Number of consecutive rows checked for causal overlap and episode padding.",
    )
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

    audit_causal_torque_windows(dataset, key, args.window_size, args.sequence_checks)

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
    if torch.isclose(torques.std(), torch.tensor(0.0), atol=1e-8):
        raise ValueError("Sampled gripper torque is constant; sensor/collector wiring may be broken")


if __name__ == "__main__":
    main()
