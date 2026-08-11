#!/usr/bin/env python3
"""High-yield visual localization data from native Factory reset randomization.

The peg is randomized and physically grasped by the environment's own reset
procedure.  Each successful episode then follows a *single downward* visual
servo trajectory; it never asks a contacted peg to retract or translate.
Object-pose values are audit-only and are excluded from training observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--dataset-file", type=Path, required=True)
p.add_argument("--num-demos", type=int, default=120)
p.add_argument("--max-attempts", type=int, default=480)
p.add_argument("--max-steps", type=int, default=420)
p.add_argument("--episode-length-s", type=float, default=30.0)
p.add_argument("--initial-up-steps", type=int, default=35)
p.add_argument("--descent-start-m", type=float, default=0.025)
p.add_argument("--descent-per-step-m", type=float, default=0.00022)
p.add_argument("--coarse-xy-action-clip", type=float, default=0.25)
p.add_argument("--fine-xy-action-clip", type=float, default=0.05)
p.add_argument("--fine-band-min-m", type=float, default=0.001)
p.add_argument("--fine-band-max-m", type=float, default=0.004)
p.add_argument("--min-fine-band-frames", type=int, default=8)
p.add_argument("--hand-noise-xy-m", type=float, default=0.006)
p.add_argument("--held-noise-xy-m", type=float, default=0.0025)
p.add_argument("--resolution", type=int, default=84)
p.add_argument("--seed", type=int, default=20260901)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not 0 < a.fine_band_min_m < a.fine_band_max_m or a.min_fine_band_frames < 1:
    p.error("invalid fine visual band")

app = AppLauncher(a).app

import h5py
import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def npv(x): return x.detach().float().cpu().numpy()


def state(e):
    return np.concatenate((npv(e.joint_pos[0, :9]), npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)


def pose_error(e):
    held, fixed = npv(e.held_pos[0]), npv(e.fixed_pos[0])
    delta = held - fixed
    return float(np.linalg.norm(delta[:2])), float(delta[2])


def strict(e):
    xy, z = pose_error(e)
    return xy < 0.0025 and z < 0.001, xy, z


def action(e, step: int, xy_error: float):
    fixed, held = npv(e.fixed_pos[0]), npv(e.held_pos[0])
    if step < a.initial_up_steps:
        target_z = 0.030
    else:
        target_z = max(-0.006, a.descent_start_m - a.descent_per_step_m * (step - a.initial_up_steps))
    target = fixed + np.array([0.0, 0.0, target_z], np.float32)
    u = np.zeros(6, np.float32)
    xy_clip = a.fine_xy_action_clip if a.fine_band_min_m <= xy_error <= a.fine_band_max_m else a.coarse_xy_action_clip
    u[:2] = np.clip((target[:2] - held[:2]) / 0.01, -xy_clip, xy_clip)
    u[2] = np.clip((target[2] - held[2]) / 0.01, -0.25, 0.25)
    return torch.from_numpy(u).to(e.device)[None]


def cameras(resolution: int):
    result = {}
    for name, pos in {"rgb_table": (1.10, 0.0, 0.80), "rgb_side": (0.65, -0.85, 0.52)}.items():
        cam = rep.create.camera(position=pos, look_at=(0.0, 0.0, 0.30))
        product = rep.create.render_product(cam, resolution=(resolution, resolution))
        product = product if isinstance(product, str) else product.path
        ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        ann.attach(product)
        result[name] = ann
    return result


def image(ann):
    x = np.asarray(ann.get_data())
    x = x[..., :3] if x.shape[-1] == 4 else x
    if x.ndim != 3 or x.shape[-1] != 3:
        raise RuntimeError(f"invalid RGB shape {x.shape}; launch with --enable_cameras")
    return x.astype(np.uint8, copy=False)


cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
cfg.seed = a.seed
cfg.episode_length_s = a.episode_length_s
# Native reset places the object, physically closes the gripper, then starts
# simulation.  These are in-gripper perturbations, not post-reset teleports.
cfg.task.hand_init_pos_noise = [a.hand_noise_xy_m, a.hand_noise_xy_m, 0.004]
cfg.task.held_asset_pos_noise = [a.held_noise_xy_m, a.held_noise_xy_m, 0.001]
cfg.sim.render_interval = cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
e = env.unwrapped
anns = cameras(a.resolution)
a.dataset_file.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(a.dataset_file, "a") as f:
    f.attrs.update(
        format="factory_peg_insert_visual_oneway_v2",
        collection_integrity="native reset randomization + pre-action observations + one-way controller trajectory",
        state_schema="proprio12:[joint_pos(9),fingertip_midpoint_pos(3)]; no peg/hole truth",
        action_schema="factory_delta_pose6:[xyz(3),axis_angle(3)]",
        audit_schema="audit_xy_error_m/audit_depth_m/is_fine_visual_band are audit-only, not policy inputs",
        episode_length_s=a.episode_length_s,
    )
    demos = f.require_group("demos")
    saved = len(demos)
    print(f"[VISUAL_ONEWAY_V2] resume={saved}/{a.num_demos} file={a.dataset_file}", flush=True)
    for attempt in range(1, a.max_attempts + 1):
        if saved >= a.num_demos: break
        episode_seed = a.seed + attempt
        env.reset(seed=episode_seed)
        torch.manual_seed(episode_seed); torch.cuda.manual_seed_all(episode_seed)
        for _ in range(3):
            env.step(torch.zeros((1, 6), dtype=torch.float32, device=e.device)); e.sim.render()
        buf = {k: [] for k in ("state", "action", "rgb_table", "rgb_side", "joint_torque", "applied_wrench", "is_fine_visual_band", "audit_xy_error_m", "audit_depth_m")}
        ok = False; xy = z = float("inf")
        for step in range(a.max_steps):
            xy, z = pose_error(e)
            u = action(e, step, xy)
            fine = a.fine_band_min_m <= xy <= a.fine_band_max_m
            buf["state"].append(state(e)); buf["action"].append(npv(u[0]))
            buf["rgb_table"].append(image(anns["rgb_table"])); buf["rgb_side"].append(image(anns["rgb_side"]))
            buf["joint_torque"].append(npv(e.joint_torque[0, :7])); buf["applied_wrench"].append(npv(e.applied_wrench[0]))
            buf["is_fine_visual_band"].append(np.array([fine], np.uint8)); buf["audit_xy_error_m"].append(np.array([xy], np.float32)); buf["audit_depth_m"].append(np.array([z], np.float32))
            env.step(u); e.sim.render()
            ok, xy, z = strict(e)
            if ok: break
        band = int(np.asarray(buf["is_fine_visual_band"], bool).sum())
        if not ok or band < a.min_fine_band_frames:
            print(f"[VISUAL_ONEWAY_V2] reject attempt={attempt} success={ok} band_frames={band} xy={xy:.5f} z={z:.5f}", flush=True)
            continue
        g = demos.create_group(f"demo_{saved:05d}")
        for key, values in buf.items(): g.create_dataset(key, data=np.stack(values), compression="gzip", compression_opts=1, shuffle=True)
        g.attrs.update(strict_success=True, frame_alignment="pre_action", state_intervention=False, attempt=attempt, episode_seed=episode_seed, fine_visual_band_frames=band, success_step=step, final_xy_error_m=xy, final_depth_m=z)
        f.flush(); saved += 1
        print(f"[VISUAL_ONEWAY_V2] saved={saved}/{a.num_demos} attempt={attempt} steps={step+1} band_frames={band} xy={xy:.5f} z={z:.5f}", flush=True)

env.close(); app.close()
if saved < a.num_demos: raise SystemExit(f"incomplete strict physical visual dataset: {saved}/{a.num_demos}")
print(f"[VISUAL_ONEWAY_V2] complete strict_demos={saved}", flush=True)
