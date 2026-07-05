#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np


REQUIRED_FIELDS = {
    "actions": (8,),
    "joint_pos": (7,),
    "gripper_pos": (2,),
    "gripper_torque": (1,),
    "rgb_table": (224, 224, 3),
    "rgb_wrist": (224, 224, 3),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-demos", type=int, default=0, help="0 audits every demo")
    args = parser.parse_args()

    if not args.path.is_file():
        raise FileNotFoundError(args.path)

    total_steps = 0
    torque_min = np.inf
    torque_max = -np.inf
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
            missing = [name for name in REQUIRED_FIELDS if name not in demo]
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
                    raise ValueError(
                        f"{demo_name}/{name}: expected [T,{trailing_shape}], got {data.shape}"
                    )

            for name in ("actions", "joint_pos", "gripper_pos", "gripper_torque"):
                values = np.asarray(demo[name])
                if not np.isfinite(values).all():
                    raise ValueError(f"{demo_name}/{name} contains NaN or Inf")

            torque = np.asarray(demo["gripper_torque"], dtype=np.float32)
            torque_min = min(torque_min, float(torque.min()))
            torque_max = max(torque_max, float(torque.max()))
            total_steps += steps

        state_schema = handle["data"].attrs.get("state_schema", handle.attrs.get("state_schema"))
        action_schema = handle["data"].attrs.get("action_schema", handle.attrs.get("action_schema"))
        fps = handle["data"].attrs.get("fps", handle.attrs.get("fps"))

    if np.isclose(torque_min, torque_max, atol=1e-8):
        raise ValueError("Gripper torque is constant across audited demonstrations")
    print(f"OK: demos={len(demos)} total_steps={total_steps}")
    print(f"OK: torque range=[{torque_min:.6g}, {torque_max:.6g}]")
    print(f"metadata: state_schema={state_schema!r} action_schema={action_schema!r} fps={fps!r}")


if __name__ == "__main__":
    main()
