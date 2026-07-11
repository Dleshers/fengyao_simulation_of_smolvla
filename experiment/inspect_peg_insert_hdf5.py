#!/usr/bin/env python3
"""Audit raw peg-insert HDF5 demonstrations before LeRobot conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


REQUIRED_FIELDS = {
    "state": (49,),
    "actions": (7,),
    "gripper_torque": (1,),
}

REQUIRED_IMAGE_FIELDS = {
    "rgb_table": 3,
    "rgb_wrist": 3,
}


def _check_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate raw peg-insert HDF5 schema produced by record_peg_insert_demos.py."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-demos", type=int, default=0, help="0 audits every demo")
    args = parser.parse_args()

    if not args.path.is_file():
        raise FileNotFoundError(args.path)

    total_steps = 0
    torque_min = np.inf
    torque_max = -np.inf
    image_shapes: dict[str, set[tuple[int, ...]]] = {key: set() for key in REQUIRED_IMAGE_FIELDS}

    with h5py.File(args.path, "r") as handle:
        if "data" not in handle:
            raise KeyError("HDF5 is missing top-level group 'data'")
        demos = sorted(handle["data"].keys())
        if not demos:
            raise ValueError("HDF5 contains no demonstrations")
        if args.max_demos > 0:
            demos = demos[: args.max_demos]

        for demo_name in demos:
            demo = handle["data"][demo_name]
            missing = [name for name in (*REQUIRED_FIELDS, *REQUIRED_IMAGE_FIELDS) if name not in demo]
            if missing:
                raise KeyError(f"{demo_name} is missing fields: {missing}")

            steps = len(demo["actions"])
            if steps == 0:
                raise ValueError(f"{demo_name} is empty")

            for name, trailing_shape in REQUIRED_FIELDS.items():
                data = demo[name]
                if len(data) != steps:
                    raise ValueError(f"{demo_name}: actions={steps}, {name}={len(data)}")
                if tuple(data.shape[1:]) != trailing_shape:
                    raise ValueError(f"{demo_name}/{name}: expected [T,{trailing_shape}], got {data.shape}")
                _check_finite(f"{demo_name}/{name}", np.asarray(data))

            for name, channels in REQUIRED_IMAGE_FIELDS.items():
                data = demo[name]
                if len(data) != steps:
                    raise ValueError(f"{demo_name}: actions={steps}, {name}={len(data)}")
                if data.ndim != 4 or data.shape[-1] != channels:
                    raise ValueError(f"{demo_name}/{name}: expected [T,H,W,{channels}], got {data.shape}")
                if data.dtype != np.uint8:
                    raise ValueError(f"{demo_name}/{name}: expected uint8 image data, got {data.dtype}")
                image_shapes[name].add(tuple(data.shape[1:]))

            torque = np.asarray(demo["gripper_torque"], dtype=np.float32)
            torque_min = min(torque_min, float(torque.min()))
            torque_max = max(torque_max, float(torque.max()))
            total_steps += steps

        state_schema = handle.attrs.get("state_schema")
        action_schema = handle.attrs.get("action_schema")
        torque_schema = handle.attrs.get("torque_schema")
        fps = handle.attrs.get("fps")

    print(f"OK: demos={len(demos)} total_steps={total_steps}")
    print(f"OK: state=[49] action=[7] gripper_torque=[1]")
    print(f"OK: torque range=[{torque_min:.6g}, {torque_max:.6g}]")
    for name, shapes in image_shapes.items():
        print(f"OK: {name} image_shapes={sorted(shapes)}")
    print(f"metadata: state_schema={state_schema!r}")
    print(f"metadata: action_schema={action_schema!r}")
    print(f"metadata: torque_schema={torque_schema!r}")
    print(f"metadata: fps={fps!r}")


if __name__ == "__main__":
    main()
