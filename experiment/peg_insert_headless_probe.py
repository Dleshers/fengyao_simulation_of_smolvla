#!/usr/bin/env python3
"""Finite headless probe for the manager-based peg insertion task.

This script is intentionally small and read-only with respect to experiment
assets.  It launches Isaac Sim through AppLauncher, creates the peg-insert
environment, prints observation/action spaces and compact observation stats,
runs a finite rollout, and exits.
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from collections.abc import Mapping

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Finite peg-insert headless probe.")
parser.add_argument("--task", type=str, default="Isaac-Peg-Insert-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=16)
parser.add_argument("--dump_after_s", type=int, default=60)
parser.add_argument(
    "--action_mode",
    choices=("zero", "small_constant", "random"),
    default="zero",
    help="Finite diagnostic action source. Keep scales small for smoke tests.",
)
parser.add_argument("--action_scale", type=float, default=0.01)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

faulthandler.enable(file=sys.stderr, all_threads=True)
if args_cli.dump_after_s > 0:
    faulthandler.dump_traceback_later(args_cli.dump_after_s, repeat=True, file=sys.stderr)

print("[PEG_PROBE] launching_app", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PEG_PROBE] app_launched", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

print("[PEG_PROBE] importing_isaaclab_tasks", flush=True)
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
print("[PEG_PROBE] imports_done", flush=True)


def _tensor_stats(value: torch.Tensor) -> str:
    value_cpu = value.detach().float().cpu()
    finite = torch.isfinite(value_cpu)
    if not bool(finite.any()):
        return f"shape={tuple(value_cpu.shape)} dtype={value.dtype} finite=0/{value_cpu.numel()}"
    finite_values = value_cpu[finite]
    return (
        f"shape={tuple(value_cpu.shape)} dtype={value.dtype} "
        f"finite={int(finite.sum())}/{value_cpu.numel()} "
        f"min={finite_values.min().item():.6g} "
        f"max={finite_values.max().item():.6g} "
        f"mean={finite_values.mean().item():.6g} "
        f"std={finite_values.std(unbiased=False).item():.6g}"
    )


def _print_obs_stats(prefix: str, obs: object) -> None:
    if isinstance(obs, torch.Tensor):
        print(f"[PEG_PROBE] {prefix}: {_tensor_stats(obs)}", flush=True)
        return
    if isinstance(obs, Mapping):
        for key, value in obs.items():
            _print_obs_stats(f"{prefix}.{key}", value)
        return
    print(f"[PEG_PROBE] {prefix}: type={type(obs).__name__}", flush=True)


def _extract_policy(obs: object) -> torch.Tensor | None:
    if isinstance(obs, Mapping):
        value = obs.get("policy")
        return value if isinstance(value, torch.Tensor) else None
    return obs if isinstance(obs, torch.Tensor) else None


def _make_action(env: gym.Env, step: int) -> torch.Tensor:
    action_shape = env.action_space.shape
    action = torch.zeros(action_shape, device=env.unwrapped.device)
    if args_cli.action_mode == "zero":
        return action
    if args_cli.action_mode == "small_constant":
        action[..., 0] = args_cli.action_scale
        if action.shape[-1] > 2:
            action[..., 2] = args_cli.action_scale * 0.5
        return action
    if args_cli.action_mode == "random":
        generator = torch.Generator(device=env.unwrapped.device)
        generator.manual_seed(1000 + step)
        return (torch.rand(action_shape, device=env.unwrapped.device, generator=generator) * 2.0 - 1.0) * args_cli.action_scale
    raise ValueError(f"Unsupported action_mode: {args_cli.action_mode}")


def main() -> None:
    print("[PEG_PROBE] parsing_env_cfg", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.enable_cameras:
        # The peg-insert visual policy only consumes RGB. On this AutoDL headless
        # stack the distance_to_image_plane annotator can return an empty buffer,
        # so keep camera sensors RGB-only for data collection/evaluation.
        for name in ("wrist_cam", "table_cam"):
            sensor_cfg = getattr(env_cfg.scene, name, None)
            if sensor_cfg is not None and hasattr(sensor_cfg, "data_types"):
                sensor_cfg.data_types = ["rgb"]
                print(f"[PEG_PROBE] camera_rgb_only={name}", flush=True)
    else:
        for name in ("wrist_cam", "table_cam"):
            if hasattr(env_cfg.scene, name):
                setattr(env_cfg.scene, name, None)
                print(f"[PEG_PROBE] disabled_scene_sensor={name}", flush=True)
        if hasattr(env_cfg.observations, "rgb_camera"):
            env_cfg.observations.rgb_camera = None
            print("[PEG_PROBE] disabled_observation_group=rgb_camera", flush=True)
    print("[PEG_PROBE] making_env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[PEG_PROBE] task={args_cli.task}", flush=True)
    print(f"[PEG_PROBE] observation_space={env.observation_space}", flush=True)
    print(f"[PEG_PROBE] action_space={env.action_space}", flush=True)
    print(
        f"[PEG_PROBE] action_mode={args_cli.action_mode} action_scale={args_cli.action_scale}",
        flush=True,
    )

    print("[PEG_PROBE] resetting", flush=True)
    obs, info = env.reset()
    print(f"[PEG_PROBE] reset_obs_type={type(obs).__name__}", flush=True)
    print(f"[PEG_PROBE] reset_info_keys={sorted(info.keys()) if isinstance(info, dict) else type(info).__name__}", flush=True)
    _print_obs_stats("reset_obs", obs)
    initial_policy = _extract_policy(obs)
    initial_eef = None
    if initial_policy is not None and initial_policy.shape[-1] >= 28:
        initial_eef = initial_policy[..., 25:28].detach().clone()
        print(f"[PEG_PROBE] reset_eef_pos={initial_eef.detach().cpu().tolist()}", flush=True)

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = _make_action(env, step)
            _obs, _rew, terminated, truncated, _info = env.step(actions)
            if step in {0, args_cli.num_steps - 1}:
                print(
                    "[PEG_PROBE] "
                    f"step={step} "
                    f"terminated={terminated.detach().cpu().tolist() if hasattr(terminated, 'detach') else terminated} "
                    f"truncated={truncated.detach().cpu().tolist() if hasattr(truncated, 'detach') else truncated}",
                    flush=True,
                )
                print(f"[PEG_PROBE] action_stats_step_{step}: {_tensor_stats(actions)}", flush=True)
                _print_obs_stats(f"obs_step_{step}", _obs)

    final_policy = _extract_policy(_obs)
    if initial_eef is not None and final_policy is not None and final_policy.shape[-1] >= 28:
        final_eef = final_policy[..., 25:28].detach().clone()
        delta = final_eef - initial_eef
        print(f"[PEG_PROBE] final_eef_pos={final_eef.detach().cpu().tolist()}", flush=True)
        print(f"[PEG_PROBE] delta_eef_pos={delta.detach().cpu().tolist()}", flush=True)

    env.close()
    print("[PEG_PROBE] success", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[PEG_PROBE] exception type={type(exc).__name__} repr={exc!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        simulation_app.close()
