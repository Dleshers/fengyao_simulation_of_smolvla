#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


REQUIRED_PREFIXES = ("torque_lstm.", "torque_norm.", "torque_to_expert.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    candidates = [args.checkpoint]
    if args.checkpoint.is_dir():
        candidates = list(args.checkpoint.rglob("*.safetensors")) + list(args.checkpoint.rglob("*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint weights found under {args.checkpoint}")

    state = None
    for path in candidates:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            loaded = load_file(path)
        else:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            state = loaded.get("model", loaded)
            if any(any(prefix in key for prefix in REQUIRED_PREFIXES) for key in state):
                break
    if state is None:
        raise ValueError("Could not read a model state dictionary")

    missing = [prefix for prefix in REQUIRED_PREFIXES if not any(prefix in key for key in state)]
    if missing:
        raise ValueError(f"Checkpoint is missing torque modules: {missing}")
    for prefix in REQUIRED_PREFIXES:
        count = sum(prefix in key for key in state)
        print(f"OK: {prefix} ({count} tensors)")


if __name__ == "__main__":
    main()
