#!/usr/bin/env python3
"""Build a torque-disambiguation diagnostic HDF5 dataset.

This script intentionally constructs paired episodes where vision and low-level
state are copied from the same source trajectory, while the gripper-torque
history and expert action differ by contact mode.  The resulting dataset is not
claimed to be a physical peg-insertion benchmark.  It is a targeted diagnostic:
a visual-only policy sees ambiguous observations, while a torque-aware policy
can resolve which recovery/proceed action is correct.

Raw output contract matches ``record_peg_insert_oracle_demos.py``:
  - state: [T,49]
  - actions: [T,7]
  - rgb_table/rgb_wrist: [T,H,W,3] uint8
  - gripper_torque: [T,1]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


REQUIRED_FIELDS = ("state", "actions", "rgb_table", "rgb_wrist", "gripper_torque")
ACTION_DIM = 7
STATE_DIM = 49


@dataclass(frozen=True)
class ContactMode:
    name: str
    mode_id: int
    torque_mean: float
    torque_amp: float
    action_xyz: tuple[float, float, float]
    gripper_cmd: float
    description: str


CONTACT_MODES = (
    ContactMode(
        name="contact_ok_proceed",
        mode_id=0,
        torque_mean=-25.0,
        torque_amp=2.5,
        action_xyz=(0.0, 0.0, -0.020),
        gripper_cmd=-1.0,
        description="Normal contact: keep grasp closed and proceed gently downward.",
    ),
    ContactMode(
        name="low_torque_slip_regrasp",
        mode_id=1,
        torque_mean=-1.0,
        torque_amp=0.35,
        action_xyz=(0.0, 0.0, 0.040),
        gripper_cmd=-1.0,
        description="Low/no contact: lift slightly while closing to recover/regrasp.",
    ),
    ContactMode(
        name="high_torque_jam_retreat",
        mode_id=2,
        torque_mean=-80.0,
        torque_amp=6.0,
        action_xyz=(0.0, 0.0, 0.060),
        gripper_cmd=1.0,
        description="Excess force/jam: retreat upward and open to release.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a force/torque-disambiguation diagnostic raw HDF5 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Source raw peg/preinsert HDF5.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output raw diagnostic HDF5.")
    parser.add_argument("--window-len", type=int, default=40, help="Frames per derived diagnostic episode.")
    parser.add_argument(
        "--anchors",
        type=float,
        nargs="+",
        default=(0.45, 0.70, 0.90),
        help="Source-trajectory fractional anchor positions used to sample ambiguous visual states.",
    )
    parser.add_argument("--max-source-demos", type=int, default=0, help="0 means use all source demos.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--keep-source-actions-prefix",
        type=int,
        default=0,
        help=(
            "If >0, keep this many initial source actions in each derived episode before switching to "
            "mode-specific diagnostic actions. Default 0 makes the whole episode torque-conditioned."
        ),
    )
    return parser.parse_args()


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _slice_with_left_pad(array: np.ndarray, end_inclusive: int, length: int) -> np.ndarray:
    start = end_inclusive - length + 1
    if start >= 0:
        return array[start : end_inclusive + 1]
    pad = np.repeat(array[:1], -start, axis=0)
    return np.concatenate([pad, array[: end_inclusive + 1]], axis=0)


def _make_torque(mode: ContactMode, length: int, rng: np.random.Generator) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float32).reshape(-1, 1)
    oscillation = mode.torque_amp * np.sin(phase)
    noise = rng.normal(loc=0.0, scale=max(mode.torque_amp * 0.10, 0.05), size=(length, 1))
    return (mode.torque_mean + oscillation + noise).astype(np.float32)


def _make_actions(source_actions: np.ndarray, mode: ContactMode, keep_prefix: int) -> np.ndarray:
    actions = np.zeros_like(source_actions, dtype=np.float32)
    actions[:, 0:3] = np.asarray(mode.action_xyz, dtype=np.float32)
    actions[:, 6] = mode.gripper_cmd
    if keep_prefix > 0:
        prefix = min(keep_prefix, actions.shape[0])
        actions[:prefix] = source_actions[:prefix]
    return actions


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")
    if args.window_len <= 0:
        raise ValueError("--window-len must be positive")
    if not all(0.0 <= anchor <= 1.0 for anchor in args.anchors):
        raise ValueError("--anchors must be fractional positions in [0,1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.input, "r") as src, h5py.File(args.output, "w") as dst:
        if "data" not in src:
            raise KeyError("Expected top-level 'data' group in source HDF5")
        _copy_attrs(src.attrs, dst.attrs)
        dst.attrs["collection_mode"] = "torque_disambiguation_diagnostic"
        dst.attrs["diagnostic_note"] = (
            "Vision/state are copied from source trajectories; torque and actions are mode-conditioned. "
            "Use this to test whether the torque-injection path can resolve intentionally ambiguous observations."
        )
        dst.attrs["source_hdf5"] = str(args.input)
        dst.attrs["window_len"] = args.window_len
        dst.attrs["anchors"] = np.asarray(args.anchors, dtype=np.float32)
        dst.attrs["contact_modes"] = ",".join(mode.name for mode in CONTACT_MODES)

        out_data = dst.create_group("data")
        source_demo_names = sorted(src["data"].keys())
        if args.max_source_demos > 0:
            source_demo_names = source_demo_names[: args.max_source_demos]
        if not source_demo_names:
            raise ValueError("No source demos found")

        out_index = 0
        total_frames = 0
        for source_demo_name in source_demo_names:
            source_demo = src["data"][source_demo_name]
            for key in REQUIRED_FIELDS:
                if key not in source_demo:
                    raise KeyError(f"{source_demo_name} is missing required field {key!r}")

            states = np.asarray(source_demo["state"], dtype=np.float32)
            actions = np.asarray(source_demo["actions"], dtype=np.float32)
            if states.ndim != 2 or states.shape[1] != STATE_DIM:
                raise ValueError(f"{source_demo_name}/state shape {states.shape}, expected [T,{STATE_DIM}]")
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(f"{source_demo_name}/actions shape {actions.shape}, expected [T,{ACTION_DIM}]")
            if states.shape[0] != actions.shape[0]:
                raise ValueError(f"{source_demo_name} state/action length mismatch")
            if states.shape[0] < 2:
                raise ValueError(f"{source_demo_name} is too short")

            length = actions.shape[0]
            anchors = sorted({min(length - 1, max(0, int(round(frac * (length - 1))))) for frac in args.anchors})
            for anchor in anchors:
                state_slice = _slice_with_left_pad(states, anchor, args.window_len)
                action_slice = _slice_with_left_pad(actions, anchor, args.window_len)
                rgb_table_slice = _slice_with_left_pad(np.asarray(source_demo["rgb_table"]), anchor, args.window_len)
                rgb_wrist_slice = _slice_with_left_pad(np.asarray(source_demo["rgb_wrist"]), anchor, args.window_len)

                for mode in CONTACT_MODES:
                    demo = out_data.create_group(f"demo_{out_index:06d}")
                    demo.attrs["source_demo"] = source_demo_name
                    demo.attrs["source_anchor"] = anchor
                    demo.attrs["contact_mode"] = mode.name
                    demo.attrs["contact_mode_id"] = mode.mode_id
                    demo.attrs["description"] = mode.description
                    demo.attrs["synthetic_action_labels"] = True
                    demo.create_dataset("state", data=state_slice.astype(np.float32), compression="gzip")
                    demo.create_dataset(
                        "actions",
                        data=_make_actions(action_slice, mode, args.keep_source_actions_prefix),
                        compression="gzip",
                    )
                    demo.create_dataset("rgb_table", data=rgb_table_slice.astype(np.uint8), compression="gzip")
                    demo.create_dataset("rgb_wrist", data=rgb_wrist_slice.astype(np.uint8), compression="gzip")
                    demo.create_dataset("gripper_torque", data=_make_torque(mode, args.window_len, rng), compression="gzip")
                    out_index += 1
                    total_frames += args.window_len

        out_data.attrs["total"] = out_index
        dst.attrs["num_demos"] = out_index
        dst.attrs["num_frames"] = total_frames

    print(f"Wrote torque-disambiguation HDF5: {args.output}")
    print(f"Source demos: {len(source_demo_names)}")
    print(f"Derived demos: {out_index}")
    print(f"Frames: {total_frames}")
    print("Modes:")
    for mode in CONTACT_MODES:
        print(f"  {mode.mode_id}: {mode.name} torque≈{mode.torque_mean} action_xyz={mode.action_xyz} grip={mode.gripper_cmd}")


if __name__ == "__main__":
    main()
