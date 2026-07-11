#!/usr/bin/env python3
"""Audit peg-insert scene poses and observation packing.

This is a finite, read-only diagnostic used before collecting demonstrations.
It compares:
  * policy observation slices,
  * Isaac Lab asset root poses,
  * relative peg-to-hole vector,
  * configured USD/procedural asset mode.

The script exits with ``os._exit(0)`` to avoid known Isaac/Kit shutdown hangs in
this workspace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Audit peg-insert pose semantics.")
parser.add_argument("--task", type=str, default="Isaac-Peg-Insert-Franka-IK-Rel-v0")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _tolist(t: torch.Tensor):
    return [round(float(x), 6) for x in t.detach().cpu().flatten().tolist()]


def _state_segments(obs):
    state = obs["policy"][0].detach().cpu()
    return {
        "shape": list(state.shape),
        "joint_pos_rel": _tolist(state[0:9]),
        "joint_vel_rel": _tolist(state[9:18]),
        "last_action": _tolist(state[18:25]),
        "eef_pos": _tolist(state[25:28]),
        "eef_quat": _tolist(state[28:32]),
        "peg_pos": _tolist(state[32:35]),
        "peg_quat": _tolist(state[35:39]),
        "hole_pos": _tolist(state[39:42]),
        "hole_quat": _tolist(state[42:46]),
        "peg_to_hole_pos": _tolist(state[46:49]),
    }


def _asset_pose(env, name: str):
    asset = env.scene[name]
    return {
        "root_pos_w": _tolist(asset.data.root_pos_w[0]),
        "root_quat_w": _tolist(asset.data.root_quat_w[0]),
        "default_root_state": _tolist(asset.data.default_root_state[0, :7]),
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    reports = []
    with torch.inference_mode():
        zero = torch.zeros((1, 7), device=env.device)
        for step in range(args_cli.steps):
            report = {
                "step": step,
                "env_origin": _tolist(env.scene.env_origins[0]),
                "robot": _asset_pose(env, "robot"),
                "peg": _asset_pose(env, "peg"),
                "hole": _asset_pose(env, "hole"),
                "policy": _state_segments(obs),
            }
            peg = torch.tensor(report["policy"]["peg_pos"])
            hole = torch.tensor(report["policy"]["hole_pos"])
            report["derived_policy_hole_minus_peg"] = _tolist(hole - peg)
            reports.append(report)
            obs, *_ = env.step(zero)

    payload = {
        "task": args_cli.task,
        "device": args_cli.device,
        "PEG_INSERT_PROCEDURAL_ASSETS": os.environ.get("PEG_INSERT_PROCEDURAL_ASSETS"),
        "PEG_INSERT_SIMPLE_TABLE": os.environ.get("PEG_INSERT_SIMPLE_TABLE"),
        "LOCAL_ISAAC_4_5_ASSET_ROOT": os.environ.get("LOCAL_ISAAC_4_5_ASSET_ROOT"),
        "reports": reports,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(text + "\n", encoding="utf-8")
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
