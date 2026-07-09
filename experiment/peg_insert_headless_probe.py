#!/usr/bin/env python3
"""Finite headless probe for the manager-based peg insertion task.

This script is intentionally tiny and read-only with respect to experiment
assets.  It launches Isaac Sim through AppLauncher, creates the peg-insert
environment, prints observation/action spaces, runs a small number of zero
action steps, and exits.
"""

from __future__ import annotations

import argparse
import faulthandler
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Finite peg-insert headless probe.")
parser.add_argument("--task", type=str, default="Isaac-Peg-Insert-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=16)
parser.add_argument("--dump_after_s", type=int, default=60)
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


def main() -> None:
    print("[PEG_PROBE] parsing_env_cfg", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print("[PEG_PROBE] making_env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[PEG_PROBE] task={args_cli.task}", flush=True)
    print(f"[PEG_PROBE] observation_space={env.observation_space}", flush=True)
    print(f"[PEG_PROBE] action_space={env.action_space}", flush=True)

    print("[PEG_PROBE] resetting", flush=True)
    obs, info = env.reset()
    print(f"[PEG_PROBE] reset_obs_type={type(obs).__name__}", flush=True)
    print(f"[PEG_PROBE] reset_info_keys={sorted(info.keys()) if isinstance(info, dict) else type(info).__name__}", flush=True)

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            _obs, _rew, terminated, truncated, _info = env.step(actions)
            if step in {0, args_cli.num_steps - 1}:
                print(
                    "[PEG_PROBE] "
                    f"step={step} "
                    f"terminated={terminated.detach().cpu().tolist() if hasattr(terminated, 'detach') else terminated} "
                    f"truncated={truncated.detach().cpu().tolist() if hasattr(truncated, 'detach') else truncated}",
                    flush=True,
                )

    env.close()
    print("[PEG_PROBE] success", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
        simulation_app.close()
