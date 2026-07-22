#!/usr/bin/env python3
"""Strict Factory peg-in-hole collection with RGB; train-state excludes object truth."""
from __future__ import annotations
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--dataset_file", type=Path, required=True)
p.add_argument("--num_demos", type=int, default=200)
p.add_argument("--max_attempts", type=int, default=300)
p.add_argument("--max_steps", type=int, default=360)
p.add_argument("--resolution", type=int, default=84)
p.add_argument("--seed", type=int, default=20260721)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import h5py
import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def npv(x): return x.detach().float().cpu().numpy()
def state(e): return np.concatenate((npv(e.joint_pos[0, :9]), npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)
def success(e):
    held, fixed = npv(e.held_pos[0]), npv(e.fixed_pos[0])
    xy, z = float(np.linalg.norm(held[:2]-fixed[:2])), float(held[2]-fixed[2])
    return xy < .0025 and z < .001, xy, z
def action(e, step):
    fixed, held = npv(e.fixed_pos[0]), npv(e.held_pos[0])
    if step < 35: target = fixed + np.array([0, 0, .030], np.float32)
    else:
        k = step - 35; r = min(.00035 + .000035*k, .0045); t = .24*k
        target = fixed + np.array([r*np.cos(t), r*np.sin(t), -.006], np.float32)
    x = np.zeros(6, np.float32); x[:3] = np.clip((target-held)/.01, -1, 1)
    return torch.from_numpy(x).to(e.device)[None]
def cameras(res):
    ans = {}
    for name, pos in {"rgb_table":(1.10,0,.80), "rgb_side":(.65,-.85,.52)}.items():
        cam = rep.create.camera(position=pos, look_at=(0,0,.30))
        prod = rep.create.render_product(cam, resolution=(res,res))
        if not isinstance(prod, str): prod = prod.path
        ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu"); ann.attach(prod); ans[name] = ann
    return ans
def image(ann):
    x = np.asarray(ann.get_data())
    if x.shape[-1] == 4: x = x[...,:3]
    if x.ndim != 3 or x.shape[-1] != 3: raise RuntimeError(f"bad RGB {x.shape}")
    return x.astype(np.uint8, copy=False)

cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
cfg.seed = a.seed; cfg.sim.render_interval = cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg); e = env.unwrapped; anns = cameras(a.resolution)
a.dataset_file.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(a.dataset_file, "a") as f:
    f.attrs["format"] = "factory_peg_insert_formal_v1"
    f.attrs["success_definition"] = "held_vs_hole: xy<0.0025m and held_z-hole_z<0.001m"
    f.attrs["state_schema"] = "proprio12:[joint_pos(9),fingertip_midpoint_pos(3)]; no peg/hole truth"
    f.attrs["action_schema"] = "factory_delta_pose6:[xyz(3),axis_angle(3)]"
    f.attrs["force_schema"] = "joint_torque7,controller_applied_wrench6"
    demos = f.require_group("demos"); saved = len(demos)
    print(f"[FORMAL_COLLECT] resume={saved}/{a.num_demos} file={a.dataset_file}", flush=True)
    for attempt in range(saved + 1, saved + a.max_attempts + 1):
        if saved >= a.num_demos: break
        env.reset(seed=a.seed+attempt)
        b = {k:[] for k in ("state","action","rgb_table","rgb_side","joint_torque","applied_wrench")}
        ok = False; xy = z = float("inf")
        for step in range(a.max_steps):
            u = action(e, step); env.step(u); e.sim.render()
            b["state"].append(state(e)); b["action"].append(npv(u[0]))
            b["rgb_table"].append(image(anns["rgb_table"])); b["rgb_side"].append(image(anns["rgb_side"]))
            b["joint_torque"].append(npv(e.joint_torque[0,:7])); b["applied_wrench"].append(npv(e.applied_wrench[0]))
            ok, xy, z = success(e)
            if ok: break
        if not ok:
            print(f"[FORMAL_COLLECT] reject attempt={attempt} xy={xy:.5f} z={z:.5f}", flush=True); continue
        g = demos.create_group(f"demo_{saved:05d}")
        for k,v in b.items(): g.create_dataset(k, data=np.stack(v), compression="gzip", compression_opts=1, shuffle=True)
        g.attrs.update(strict_success=True, success_step=step, final_xy_error_m=xy, final_depth_m=z, attempt=attempt)
        f.flush(); saved += 1
        print(f"[FORMAL_COLLECT] saved={saved}/{a.num_demos} attempt={attempt} steps={step+1} xy={xy:.5f} z={z:.5f}", flush=True)
env.close(); app.close()
if saved < a.num_demos: raise SystemExit(f"only {saved}/{a.num_demos} strict demos")
print(f"[FORMAL_COLLECT] complete strict_demos={saved}", flush=True)
