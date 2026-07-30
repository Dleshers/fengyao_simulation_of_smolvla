#!/usr/bin/env python3
"""Collect strict insertion plus audited, conditional near-rim recovery demos.

Recovery starts are controlled *state interventions*: after IK places the hand near
the hole, the grasped peg is put at a measured lateral/depth error.  The
intervention itself is never a labelled action.  Only corrective oracle actions
and their pre-action RGB/proprioception/7D torque observations are stored.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser()
p.add_argument("--dataset-file", type=Path, required=True)
p.add_argument("--normal-demos", type=int, default=100)
p.add_argument("--per-stratum", type=int, default=100)
p.add_argument("--max-attempts", type=int, default=1800)
p.add_argument("--normal-max-steps", type=int, default=360)
p.add_argument("--recovery-max-steps", type=int, default=220)
p.add_argument("--resolution", type=int, default=84)
p.add_argument("--initial-depth", type=float, default=0.003)
p.add_argument("--seed", type=int, default=20260730)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.normal_demos < 0 or a.per_stratum < 1 or a.initial_depth <= 0.001:
    p.error("invalid target counts or initial depth")
app = AppLauncher(a).app

import h5py
import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


STRATA = {
    "easy": {"radii": (0.0040, 0.0044, 0.0048, 0.0050), "min_xy": 0.0025, "max_xy": 0.0045},
    "medium": {"radii": (0.0050, 0.0053, 0.0057, 0.0060), "min_xy": 0.0045, "max_xy": 0.0060},
    "hard": {"radii": (0.0062, 0.0066, 0.0070, 0.0074), "min_xy": 0.0060, "max_xy": 0.0075},
}


def npv(x):
    return x.detach().float().cpu().numpy()


def state(e):
    return np.concatenate((npv(e.joint_pos[0, :9]), npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)


def pose_error(e):
    held, fixed = npv(e.held_pos[0]), npv(e.fixed_pos[0])
    return float(np.linalg.norm(held[:2] - fixed[:2])), float(held[2] - fixed[2])


def strict(e):
    xy, z = pose_error(e)
    return xy < 0.0025 and z < 0.001, xy, z


def normal_oracle(e, step):
    fixed, held = npv(e.fixed_pos[0]), npv(e.held_pos[0])
    if step < 35:
        target = fixed + np.array([0, 0, 0.030], np.float32)
    else:
        k = step - 35
        radius = min(0.00035 + 0.000035 * k, 0.0045)
        theta = 0.24 * k
        target = fixed + np.array([radius * np.cos(theta), radius * np.sin(theta), -0.006], np.float32)
    u = np.zeros(6, np.float32)
    u[:3] = np.clip((target - held) / 0.01, -1, 1)
    return torch.from_numpy(u).to(e.device)[None]


def recovery_oracle(e):
    """Directly recenter while descending; valid only after a near-rim intervention."""
    fixed, held = npv(e.fixed_pos[0]), npv(e.held_pos[0])
    target = fixed + np.array([0, 0, -0.006], np.float32)
    u = np.zeros(6, np.float32)
    u[:3] = np.clip((target - held) / 0.01, -1, 1)
    return torch.from_numpy(u).to(e.device)[None]


def cameras(res):
    out = {}
    for name, pos in {"rgb_table": (1.10, 0, .80), "rgb_side": (.65, -.85, .52)}.items():
        cam = rep.create.camera(position=pos, look_at=(0, 0, .30))
        product = rep.create.render_product(cam, resolution=(res, res))
        product = product if isinstance(product, str) else product.path
        ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        ann.attach(product)
        out[name] = ann
    return out


def image(ann):
    x = np.asarray(ann.get_data())
    x = x[..., :3] if x.shape[-1] == 4 else x
    if x.ndim != 3 or x.shape[-1] != 3:
        raise RuntimeError(f"bad RGB {x.shape}")
    return x.astype(np.uint8, copy=False)


def set_near_rim_state(e, radius, angle):
    """IK the hand, then set a reproducible off-centre grasped-peg state."""
    fixed_pose = e._fixed_asset.data.root_pose_w.clone()
    held_before = e._held_asset.data.root_pose_w.clone()
    hand_pos, hand_quat = e.fingertip_midpoint_pos.clone(), e.fingertip_midpoint_quat.clone()
    hand_minus_held = hand_pos - held_before[:, :3]
    target_held = fixed_pose.clone()
    target_held[:, 0] += float(radius * np.cos(angle))
    target_held[:, 1] += float(radius * np.sin(angle))
    target_held[:, 2] += a.initial_depth
    target_held[:, 3:7] = held_before[:, 3:7]
    target_hand = target_held[:, :3] + hand_minus_held
    env_ids = torch.arange(e.num_envs, device=e.device, dtype=torch.long)
    e.set_pos_inverse_kinematics(target_hand, hand_quat, env_ids)
    zeros = torch.zeros((e.num_envs, 6), device=e.device)
    e._held_asset.write_root_pose_to_sim(target_held[:, :7])
    e._held_asset.write_root_velocity_to_sim(zeros)
    e._held_asset.reset()
    e.actions.zero_()
    e.ctrl_target_joint_pos.copy_(e.joint_pos)
    e._robot.set_joint_position_target(e.ctrl_target_joint_pos)
    e.step_sim_no_action()
    e._compute_intermediate_values(dt=e.physics_dt)
    return pose_error(e)


def append_frame(buf, e, anns, u, recovery):
    buf["state"].append(state(e))
    buf["action"].append(npv(u[0]))
    buf["rgb_table"].append(image(anns["rgb_table"]))
    buf["rgb_side"].append(image(anns["rgb_side"]))
    buf["joint_torque"].append(npv(e.joint_torque[0, :7]))
    buf["applied_wrench"].append(npv(e.applied_wrench[0]))
    buf["is_recovery"].append(np.array([recovery], np.uint8))


cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
cfg.seed = a.seed
cfg.sim.render_interval = cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
e = env.unwrapped
anns = cameras(a.resolution)
a.dataset_file.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(a.dataset_file, "a") as f:
    f.attrs.update(
        format="factory_peg_insert_conditional_recovery_v3",
        success_definition="held_vs_hole: xy<0.0025m and held_z-hole_z<0.001m",
        observation_action_alignment="state/rgb/joint_torque observed before action_t; action_t executed after recording",
        state_schema="proprio12:[joint_pos(9),fingertip_midpoint_pos(3)]; no peg/hole truth",
        action_schema="factory_delta_pose6:[xyz(3),axis_angle(3)]",
        force_schema="joint_torque7,controller_applied_wrench6",
        recovery_policy="deterministic hand-IK plus peg state intervention followed by oracle-labelled corrective actions",
    )
    demos = f.require_group("demos")
    counts = {"normal": 0, **{k: 0 for k in STRATA}}
    for name in demos:
        kind = demos[name].attrs.get("difficulty_stratum", "normal")
        if isinstance(kind, bytes):
            kind = kind.decode()
        if kind in counts:
            counts[kind] += 1
    target_total = a.normal_demos + len(STRATA) * a.per_stratum
    print(f"[CONDITIONAL_RECOVERY] resume counts={counts} target={target_total} file={a.dataset_file}", flush=True)
    for attempt in range(1, a.max_attempts + 1):
        pending = (["normal"] if counts["normal"] < a.normal_demos else []) + [k for k in STRATA if counts[k] < a.per_stratum]
        if not pending:
            break
        kind = pending[(attempt - 1) % len(pending)]
        rng = np.random.default_rng(a.seed + attempt * 7919)
        env.reset(seed=a.seed + attempt)
        env.step(torch.zeros((1, 6), dtype=torch.float32, device=e.device))
        e.sim.render()
        angle = float(rng.uniform(-np.pi, np.pi))
        initial_xy = initial_z = None
        if kind != "normal":
            # Deterministic state intervention: first move the hand by Factory IK,
            # then set the grasped peg and accept only the measured near-rim state.
            spec = STRATA[kind]
            requested_radius = float(rng.choice(spec["radii"]))
            initial_xy, initial_z = set_near_rim_state(e, requested_radius, angle)
            valid = spec["min_xy"] <= initial_xy < spec["max_xy"] and 0.001 < initial_z <= 0.008
            if not valid:
                print(f"[CONDITIONAL_RECOVERY] reject state-init attempt={attempt} stratum={kind} requested_r={requested_radius:.4f} xy={initial_xy:.5f} z={initial_z:.5f}", flush=True)
                continue
        buf = {k: [] for k in ("state", "action", "rgb_table", "rgb_side", "joint_torque", "applied_wrench", "is_recovery")}
        max_steps = a.normal_max_steps if kind == "normal" else a.recovery_max_steps
        ok = False
        xy = z = float("inf")
        for step in range(max_steps):
            u = normal_oracle(e, step if kind == "normal" else 35 + step)
            append_frame(buf, e, anns, u, kind != "normal")
            env.step(u)
            e.sim.render()
            ok, xy, z = strict(e)
            if ok:
                break
        if not ok:
            print(f"[CONDITIONAL_RECOVERY] reject rollout attempt={attempt} stratum={kind} xy={xy:.5f} z={z:.5f}", flush=True)
            continue
        idx = len(demos)
        g = demos.create_group(f"demo_{idx:05d}")
        for key, values in buf.items():
            g.create_dataset(key, data=np.stack(values), compression="gzip", compression_opts=1, shuffle=True)
        g.attrs.update(
            strict_success=True, success_step=step, final_xy_error_m=xy, final_depth_m=z,
            attempt=attempt, recovery_episode=kind != "normal", difficulty_stratum=kind,
            initial_xy_error_m=-1.0 if initial_xy is None else initial_xy,
            initial_depth_m=-1.0 if initial_z is None else initial_z,
            state_intervention=kind != "normal", perturbation_type="none" if kind == "normal" else "deterministic_hand_peg_state", frame_alignment="pre_action",
        )
        f.flush()
        counts[kind] += 1
        print(f"[CONDITIONAL_RECOVERY] saved={idx + 1}/{target_total} stratum={kind} counts={counts} frames={len(buf['action'])} init_xy={initial_xy} final_xy={xy:.5f} z={z:.5f}", flush=True)
    print(f"[CONDITIONAL_RECOVERY] complete counts={counts} target_total={target_total}", flush=True)
    if counts["normal"] != a.normal_demos or any(counts[k] != a.per_stratum for k in STRATA):
        raise SystemExit(f"incomplete counts={counts}")
env.close()
app.close()
