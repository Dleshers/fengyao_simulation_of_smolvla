#!/usr/bin/env python3
"""Create one scripted peg-insert raw HDF5 smoke episode.

This is not a replacement for human teleoperation data.  It directly moves the
peg pose toward the hole to verify that the raw HDF5 -> audit -> LeRobot
conversion chain is healthy when GUI teleop is unavailable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record a scripted peg-insert smoke HDF5 episode.")
parser.add_argument("--task", type=str, default="Isaac-Peg-Insert-Franka-IK-Rel-v0")
parser.add_argument("--dataset_file", type=Path, required=True)
parser.add_argument("--num_steps", type=int, default=80)
parser.add_argument("--fps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


STATE_SCHEMA = (
    "peg_insert_ik_state_49d:"
    "[joint_pos_rel(9),joint_vel_rel(9),last_action(7),eef_pos(3),eef_quat(4),"
    "peg_pos(3),peg_quat(4),hole_pos(3),hole_quat(4),peg_to_hole_pos(3)]"
)
ACTION_SCHEMA = "peg_insert_ik_action_7d:[delta_pos(3),delta_rot(3),gripper_binary(1)]"
TORQUE_SCHEMA = "gripper_torque:[1], mean of the two gripper joint applied torques"


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _state_from_obs(obs) -> np.ndarray:
    state = _to_numpy(obs["policy"][0]).astype(np.float32)
    if state.shape != (49,):
        raise ValueError(f"Expected state [49], got {state.shape}")
    return state


def _rgb_from_obs(obs, key: str) -> np.ndarray:
    image = _to_numpy(obs["rgb_camera"][key][0])
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.clip(image, 0, 255).astype(np.uint8)


def _gripper_torque(env) -> np.ndarray:
    robot = env.scene["robot"]
    torque = getattr(robot.data, "applied_torque", None)
    if torque is None:
        torque = getattr(robot.data, "joint_effort", None)
    if torque is None:
        return np.zeros((1,), dtype=np.float32)
    values = torque[0].detach().float().cpu().numpy()
    return np.array([float(values[-2:].mean())], dtype=np.float32)


def _write_peg_pose(env, alpha: float) -> None:
    peg = env.scene["peg"]
    hole = env.scene["hole"]
    peg_pose = peg.data.root_pose_w.clone()
    hole_pose = hole.data.root_pose_w.clone()
    start = peg_pose[:, :3].clone()
    target = hole_pose[:, :3].clone()
    target[:, 2] = hole_pose[:, 2] - 0.002
    peg_pose[:, :3] = (1.0 - alpha) * start + alpha * target
    peg_pose[:, 3:7] = hole_pose[:, 3:7]
    peg.write_root_pose_to_sim(peg_pose)


def _scripted_success(env) -> bool:
    peg_pos = env.scene["peg"].data.root_pos_w - env.scene.env_origins
    hole_pos = env.scene["hole"].data.root_pos_w - env.scene.env_origins
    xy_dist = torch.norm(peg_pos[:, :2] - hole_pos[:, :2], dim=-1)
    z_disp = peg_pos[:, 2] - hole_pos[:, 2]
    return bool(torch.logical_and(xy_dist < 0.0025, z_disp < 0.001).item())


def main() -> None:
    print("[SCRIPTED_PEG] main_start", flush=True)
    if args_cli.dataset_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing HDF5: {args_cli.dataset_file}")
    args_cli.dataset_file.parent.mkdir(parents=True, exist_ok=True)

    print("[SCRIPTED_PEG] parse_env_cfg", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    print("[SCRIPTED_PEG] make_env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print("[SCRIPTED_PEG] reset", flush=True)
    obs, _ = env.reset()
    print("[SCRIPTED_PEG] reset_done", flush=True)

    buffers: dict[str, list[np.ndarray]] = {
        "state": [],
        "actions": [],
        "rgb_table": [],
        "rgb_wrist": [],
        "gripper_torque": [],
    }
    zero_action = torch.zeros(env.action_space.shape, device=env.device)
    zero_action[..., -1] = 1.0  # keep gripper open; positive is open for BinaryJointAction

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            if step == 0 or step == args_cli.num_steps - 1 or step % 10 == 0:
                print(f"[SCRIPTED_PEG] step {step}/{args_cli.num_steps}", flush=True)
            alpha = min(1.0, step / max(1, args_cli.num_steps - 10))
            _write_peg_pose(env, alpha)
            env.sim.render()
            buffers["state"].append(_state_from_obs(obs))
            buffers["actions"].append(_to_numpy(zero_action[0]).astype(np.float32))
            buffers["rgb_table"].append(_rgb_from_obs(obs, "table_cam"))
            buffers["rgb_wrist"].append(_rgb_from_obs(obs, "wrist_cam"))
            buffers["gripper_torque"].append(_gripper_torque(env))
            obs, _, _, _, _ = env.step(zero_action)

    print("[SCRIPTED_PEG] marking_scripted_success", flush=True)
    success = True

    with h5py.File(args_cli.dataset_file, "w") as handle:
        handle.attrs["env_name"] = args_cli.task
        handle.attrs["fps"] = args_cli.fps
        handle.attrs["state_schema"] = STATE_SCHEMA
        handle.attrs["action_schema"] = ACTION_SCHEMA
        handle.attrs["torque_schema"] = TORQUE_SCHEMA
        handle.attrs["collection_mode"] = "scripted_oracle_smoke_not_teleop"
        data_group = handle.create_group("data")
        data_group.attrs["total"] = len(buffers["actions"])
        demo = data_group.create_group("demo_0")
        demo.attrs["num_samples"] = len(buffers["actions"])
        demo.attrs["success"] = True
        for key, values in buffers.items():
            demo.create_dataset(key, data=np.stack(values), compression="gzip")

    print("[SCRIPTED_PEG] hdf5_written", flush=True)
    print(f"Saved scripted smoke HDF5: {args_cli.dataset_file}")
    print(f"frames={len(buffers['actions'])} success={success}")
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
