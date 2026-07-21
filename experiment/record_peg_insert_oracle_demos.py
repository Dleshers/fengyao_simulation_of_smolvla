#!/usr/bin/env python3
"""Collect peg-insert demonstrations with a scripted robot-action oracle.

Unlike ``record_peg_insert_scripted_smoke.py``, this script does not teleport
the peg.  It records the actual relative-IK actions sent to the robot, so the
resulting HDF5 is suitable for behavioral-cloning pipeline validation.

The oracle is deliberately simple and conservative:
  1. open gripper and move above the peg,
  2. descend to the peg,
  3. close gripper,
  4. lift,
  5. move above the hole,
  6. descend/insert.

Only episodes whose observed peg/hole state satisfies the insertion predicate
are exported.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record peg-insert scripted/oracle robot-action demos.")
parser.add_argument("--task", type=str, default="Isaac-Peg-Insert-Franka-IK-Rel-v0")
parser.add_argument("--dataset_file", type=Path, required=True)
parser.add_argument("--num_demos", type=int, default=1)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--max_attempts", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=260)
parser.add_argument("--success_hold_steps", type=int, default=5)
parser.add_argument(
    "--success_mode",
    choices=("inserted", "preinsert_alignment"),
    default="inserted",
    help=(
        "inserted uses the strict legacy root-depth predicate. "
        "preinsert_alignment is for procedural target-block data: peg is grasped, centered above the target, "
        "and lowered close to the hole top without forcing it through a solid cuboid."
    ),
)
parser.add_argument("--pos_gain", type=float, default=2.0)
parser.add_argument("--max_xyz_action", type=float, default=0.08)
parser.add_argument(
    "--allow_missing_images",
    action="store_true",
    help="Fill RGB observations with zeros when cameras are disabled. Debug only; do not use as visual training data.",
)
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


def _np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _state(obs) -> np.ndarray:
    state = _np(obs["policy"][0]).astype(np.float32)
    if state.shape != (49,):
        raise ValueError(f"Expected state [49], got {state.shape}")
    return state


def _rgb(obs, key: str) -> np.ndarray:
    if "rgb_camera" not in obs:
        if args_cli.allow_missing_images:
            return np.zeros((84, 84, 3), dtype=np.uint8)
        raise KeyError("Missing rgb_camera observations; enable cameras or pass --allow_missing_images for debug only.")
    image = _np(obs["rgb_camera"][key][0])
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.clip(image, 0, 255).astype(np.uint8)


def _torque(env) -> np.ndarray:
    robot = env.scene["robot"]
    torque = getattr(robot.data, "applied_torque", None)
    if torque is None:
        torque = getattr(robot.data, "joint_effort", None)
    if torque is None:
        return np.zeros((1,), dtype=np.float32)
    values = torque[0].detach().float().cpu().numpy()
    return np.array([float(values[-2:].mean())], dtype=np.float32)


def _unpack(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eef = state[25:28]
    peg = state[32:35]
    hole = state[39:42]
    return eef, peg, hole


def _inserted(state: np.ndarray) -> bool:
    _eef, peg, hole = _unpack(state)
    xy_dist = float(np.linalg.norm(peg[:2] - hole[:2]))
    z_disp = float(peg[2] - hole[2])
    return xy_dist < 0.0025 and z_disp < 0.001


def _preinsert_aligned(state: np.ndarray) -> bool:
    _eef, peg, hole = _unpack(state)
    xy_dist = float(np.linalg.norm(peg[:2] - hole[:2]))
    z_disp = float(peg[2] - hole[2])
    # For the procedural cylinder peg, root_pos is near the peg center.  This
    # band corresponds to a grasped peg centered above the target and lowered
    # close to the target top, before it is pushed into the solid cuboid stand-in.
    return xy_dist < 0.005 and 0.025 < z_disp < 0.05


def _success(state: np.ndarray) -> bool:
    if args_cli.success_mode == "inserted":
        return _inserted(state)
    if args_cli.success_mode == "preinsert_alignment":
        return _preinsert_aligned(state)
    raise ValueError(f"Unsupported success_mode: {args_cli.success_mode}")


def _empty_buffers() -> dict[str, list[np.ndarray]]:
    return {"state": [], "actions": [], "rgb_table": [], "rgb_wrist": [], "gripper_torque": []}


def _action_to_target(env, state: np.ndarray, target: np.ndarray, gripper_open: bool) -> torch.Tensor:
    eef, _peg, _hole = _unpack(state)
    delta = (target - eef) * args_cli.pos_gain
    delta = np.clip(delta, -args_cli.max_xyz_action, args_cli.max_xyz_action)
    action_np = np.zeros((7,), dtype=np.float32)
    action_np[:3] = delta
    action_np[6] = 1.0 if gripper_open else -1.0
    return torch.tensor(action_np, dtype=torch.float32, device=env.device).reshape(1, 7)


def _phase_target(
    step: int, state: np.ndarray, peg_anchor: np.ndarray, hole_anchor: np.ndarray
) -> tuple[np.ndarray, bool]:
    _eef, _peg, _hole = _unpack(state)
    peg = peg_anchor
    hole = hole_anchor
    peg_above = peg + np.array([0.0, 0.0, 0.13], dtype=np.float32)
    # The IK action controls the TCP-like ee_frame located between the fingers.
    # For a vertical 8 mm peg, targeting too high above the root closes the
    # gripper above the peg and simply pushes it around.  Aim near the peg's
    # midline/top transition instead.
    peg_grasp = peg + np.array([0.0, 0.0, 0.018], dtype=np.float32)
    lift = peg + np.array([0.0, 0.0, 0.18], dtype=np.float32)
    hole_above = hole + np.array([0.0, 0.0, 0.16], dtype=np.float32)
    hole_preinsert = hole + np.array([0.0, 0.0, 0.055], dtype=np.float32)
    hole_insert = hole + np.array([0.0, 0.0, 0.018], dtype=np.float32)

    if step < 35:
        return peg_above, True
    if step < 70:
        return peg_grasp, True
    if step < 100:
        return peg_grasp, False
    if step < 125:
        return lift, False
    if step < 165:
        return hole_above, False
    if step < 205:
        return hole_preinsert, False
    return hole_insert, False


def _record_step(buffers: dict[str, list[np.ndarray]], env, obs, action: torch.Tensor) -> None:
    buffers["state"].append(_state(obs))
    buffers["actions"].append(_np(action[0]).astype(np.float32))
    buffers["rgb_table"].append(_rgb(obs, "table_cam"))
    buffers["rgb_wrist"].append(_rgb(obs, "wrist_cam"))
    buffers["gripper_torque"].append(_torque(env))


def _write_demo(group, name: str, buffers: dict[str, list[np.ndarray]]) -> int:
    steps = len(buffers["actions"])
    demo = group.create_group(name)
    demo.attrs["num_samples"] = steps
    demo.attrs["success"] = True
    demo.attrs["success_mode"] = args_cli.success_mode
    demo.attrs["collection_mode"] = "scripted_oracle_robot_action"
    for key, values in buffers.items():
        demo.create_dataset(key, data=np.stack(values), compression="gzip")
    return steps


def main() -> None:
    if args_cli.dataset_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing HDF5: {args_cli.dataset_file}")
    args_cli.dataset_file.parent.mkdir(parents=True, exist_ok=True)

    print("[PEG_ORACLE] parse_env_cfg", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if args_cli.enable_cameras:
        # The dataset only stores RGB. Avoid initializing unused depth annotators
        # that are unreliable in the no-viewport AutoDL headless path.
        for name in ("wrist_cam", "table_cam"):
            sensor_cfg = getattr(env_cfg.scene, name, None)
            if sensor_cfg is not None and hasattr(sensor_cfg, "data_types"):
                sensor_cfg.data_types = ["rgb"]
                print(f"[PEG_ORACLE] camera_rgb_only={name}", flush=True)
    else:
        for name in ("wrist_cam", "table_cam"):
            if hasattr(env_cfg.scene, name):
                setattr(env_cfg.scene, name, None)
                print(f"[PEG_ORACLE] disabled_scene_sensor={name}", flush=True)
        if hasattr(env_cfg.observations, "rgb_camera"):
            env_cfg.observations.rgb_camera = None
            print("[PEG_ORACLE] disabled_observation_group=rgb_camera", flush=True)
    print("[PEG_ORACLE] make_env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    total_steps = 0
    saved = 0
    with h5py.File(args_cli.dataset_file, "w") as handle:
        handle.attrs["env_name"] = args_cli.task
        handle.attrs["fps"] = args_cli.fps
        handle.attrs["state_schema"] = STATE_SCHEMA
        handle.attrs["action_schema"] = ACTION_SCHEMA
        handle.attrs["torque_schema"] = TORQUE_SCHEMA
        handle.attrs["collection_mode"] = "scripted_oracle_robot_action"
        handle.attrs["success_mode"] = args_cli.success_mode
        data_group = handle.create_group("data")
        data_group.attrs["total"] = 0

        with torch.inference_mode():
            for attempt in range(args_cli.max_attempts):
                if saved >= args_cli.num_demos:
                    break
                print(f"[PEG_ORACLE] attempt {attempt + 1}/{args_cli.max_attempts}", flush=True)
                obs, _ = env.reset()
                initial_state = _state(obs)
                _initial_eef, peg_anchor, hole_anchor = _unpack(initial_state)
                peg_anchor = peg_anchor.copy()
                hole_anchor = hole_anchor.copy()
                buffers = _empty_buffers()
                hold = 0
                for step in range(args_cli.max_steps):
                    state = _state(obs)
                    target, gripper_open = _phase_target(step, state, peg_anchor, hole_anchor)
                    action = _action_to_target(env, state, target, gripper_open)
                    _record_step(buffers, env, obs, action)
                    obs, _reward, terminated, truncated, _info = env.step(action)
                    next_state = _state(obs)
                    hold = hold + 1 if _success(next_state) else 0
                    if step % 25 == 0:
                        eef, peg, hole = _unpack(next_state)
                        dist = float(np.linalg.norm(eef - target))
                        print(
                            f"[PEG_ORACLE] step={step} eef={eef.round(4).tolist()} "
                            f"target={target.round(4).tolist()} dist={dist:.4f} "
                            f"peg={peg.round(4).tolist()} hole={hole.round(4).tolist()} hold={hold}",
                            flush=True,
                        )
                    if hold >= args_cli.success_hold_steps:
                        name = f"demo_{saved}"
                        steps = _write_demo(data_group, name, buffers)
                        total_steps += steps
                        data_group.attrs["total"] = total_steps
                        handle.flush()
                        saved += 1
                        print(f"[PEG_ORACLE] saved {name}: steps={steps}", flush=True)
                        break
                    if bool(terminated[0]) or bool(truncated[0]):
                        break

    print(f"[PEG_ORACLE] complete saved={saved} total_steps={total_steps}", flush=True)
    if saved < args_cli.num_demos:
        print(f"[PEG_ORACLE] ERROR only collected {saved}/{args_cli.num_demos} successful demos", flush=True)
        os._exit(2)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
