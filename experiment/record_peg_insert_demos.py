#!/usr/bin/env python3
"""Record peg-insert teleop demonstrations into the project raw HDF5 contract.

This is intentionally separate from IsaacLab's generic ``record_demos.py``:
the generic recorder stores the policy observation group but not the separate
``rgb_camera`` group, while SmolVLA training needs synchronized RGB frames.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record peg-insert teleop demonstrations with RGB and gripper torque.")
parser.add_argument("--task", type=str, default="Isaac-Peg-Insert-Franka-IK-Rel-v0")
parser.add_argument("--teleop_device", type=str, default="keyboard")
parser.add_argument("--dataset_file", type=Path, required=True)
parser.add_argument("--step_hz", type=int, default=20)
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--num_success_steps", type=int, default=10)
parser.add_argument("--max_steps_per_demo", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg  # noqa: E402
from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


STATE_SCHEMA = (
    "peg_insert_ik_state_49d:"
    "[joint_pos_rel(9),joint_vel_rel(9),last_action(7),eef_pos(3),eef_quat(4),"
    "peg_pos(3),peg_quat(4),hole_pos(3),hole_quat(4),peg_to_hole_pos(3)]"
)
ACTION_SCHEMA = "peg_insert_ik_action_7d:[delta_pos(3),delta_rot(3),gripper_binary(1)]"
TORQUE_SCHEMA = "gripper_torque:[1], mean of the two gripper joint applied torques"


class RateLimiter:
    def __init__(self, hz: int):
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)
        self.last_time = time.time()

    def sleep(self, env):
        next_time = self.last_time + self.sleep_duration
        while time.time() < next_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = max(next_time, time.time())


def _setup_teleop(env_cfg):
    callbacks = {}
    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
    if args_cli.teleop_device.lower() == "keyboard":
        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05, sim_device=args_cli.device))
    if args_cli.teleop_device.lower() == "spacemouse":
        return Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.05, rot_sensitivity=0.05, sim_device=args_cli.device))
    raise ValueError(f"Unsupported teleop_device={args_cli.teleop_device!r}")


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _state_from_obs(obs) -> np.ndarray:
    policy = obs["policy"] if isinstance(obs, dict) else obs
    if not isinstance(policy, torch.Tensor):
        raise TypeError(f"Expected concatenated policy tensor, got {type(policy).__name__}")
    state = _to_numpy(policy[0]).astype(np.float32)
    if state.shape != (49,):
        raise ValueError(f"Expected policy state [49], got {state.shape}")
    return state


def _rgb_from_obs(obs, key: str) -> np.ndarray:
    rgb = obs["rgb_camera"][key]
    image = _to_numpy(rgb[0])
    if image.ndim != 3:
        raise ValueError(f"{key} expected HWC image, got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"{key} expected RGB image, got {image.shape}")
    return np.clip(image, 0, 255).astype(np.uint8)


def _gripper_torque(env) -> np.ndarray:
    robot = env.scene["robot"]
    torque = getattr(robot.data, "applied_torque", None)
    if torque is None:
        torque = getattr(robot.data, "joint_effort", None)
    if torque is None:
        raise AttributeError("robot.data has neither applied_torque nor joint_effort")
    values = torque[0].detach().float().cpu().numpy()
    if values.shape[0] < 2:
        raise ValueError(f"Expected at least two gripper joints, got torque shape {values.shape}")
    return np.array([float(values[-2:].mean())], dtype=np.float32)


def _write_demo(data_group, name: str, buffers: dict[str, list[np.ndarray]]) -> None:
    demo = data_group.create_group(name)
    steps = len(buffers["actions"])
    demo.attrs["num_samples"] = steps
    demo.attrs["success"] = True
    for key, values in buffers.items():
        demo.create_dataset(key, data=np.stack(values), compression="gzip")


def _empty_buffers() -> dict[str, list[np.ndarray]]:
    return {"state": [], "actions": [], "rgb_table": [], "rgb_wrist": [], "gripper_torque": []}


def main() -> None:
    if args_cli.dataset_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing HDF5: {args_cli.dataset_file}")
    args_cli.dataset_file.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    success_term = env_cfg.terminations.success if hasattr(env_cfg.terminations, "success") else None
    if success_term is not None:
        env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    teleop = _setup_teleop(env_cfg)
    limiter = RateLimiter(args_cli.step_hz)

    with h5py.File(args_cli.dataset_file, "w") as handle:
        handle.attrs["env_name"] = args_cli.task
        handle.attrs["fps"] = args_cli.step_hz
        handle.attrs["state_schema"] = STATE_SCHEMA
        handle.attrs["action_schema"] = ACTION_SCHEMA
        handle.attrs["torque_schema"] = TORQUE_SCHEMA
        handle.attrs["camera_schema"] = "rgb_table/rgb_wrist:uint8[H,W,3], recorded from rgb_camera table_cam/wrist_cam"
        data_group = handle.create_group("data")
        data_group.attrs["total"] = 0

        env.sim.reset()
        obs, _ = env.reset()
        teleop.reset()

        demo_index = 0
        success_count = 0
        buffers = _empty_buffers()
        print(f"Recording peg-insert demos to {args_cli.dataset_file}")
        print("Only successful episodes are exported. Press Ctrl-C to stop safely.")

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running() and demo_index < args_cli.num_demos:
                action = teleop.advance().repeat(env.num_envs, 1)
                if action.shape[-1] != 7:
                    raise ValueError(f"Expected teleop action [7], got {tuple(action.shape)}")

                buffers["state"].append(_state_from_obs(obs))
                buffers["actions"].append(_to_numpy(action[0]).astype(np.float32))
                buffers["rgb_table"].append(_rgb_from_obs(obs, "table_cam"))
                buffers["rgb_wrist"].append(_rgb_from_obs(obs, "wrist_cam"))
                buffers["gripper_torque"].append(_gripper_torque(env))

                obs, _, _, _, _ = env.step(action)
                inserted = False
                if success_term is not None:
                    inserted = bool(success_term.func(env, **success_term.params)[0])
                success_count = success_count + 1 if inserted else 0

                if success_count >= args_cli.num_success_steps:
                    name = f"demo_{demo_index}"
                    _write_demo(data_group, name, buffers)
                    steps = len(buffers["actions"])
                    data_group.attrs["total"] += steps
                    demo_index += 1
                    handle.flush()
                    print(f"Saved {name}: {steps} steps ({demo_index}/{args_cli.num_demos})")
                    buffers = _empty_buffers()
                    success_count = 0
                    env.sim.reset()
                    obs, _ = env.reset()
                    teleop.reset()
                    continue

                if len(buffers["actions"]) >= args_cli.max_steps_per_demo:
                    print(f"Discarding unsuccessful attempt after {args_cli.max_steps_per_demo} steps")
                    buffers = _empty_buffers()
                    success_count = 0
                    env.sim.reset()
                    obs, _ = env.reset()
                    teleop.reset()

                limiter.sleep(env)

    env.close()
    print(f"Done. Successful demos recorded: {demo_index}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
