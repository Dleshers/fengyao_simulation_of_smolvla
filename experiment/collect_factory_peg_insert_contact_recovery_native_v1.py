#!/usr/bin/env python3
"""Collect native-reset, physical near-rim contact and recovery trajectories.

There is no post-reset pose write or teleport.  A trajectory approaches a
small, off-centre target through the normal Cartesian controller, holds a
downward command long enough to obtain a chronological torque history, then
unloads, recentres and inserts.  Only post-contact recovery frames carry
policy labels; the approach/contact history is retained for audit and LSTM
history construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--dataset-file", type=Path, required=True)
p.add_argument("--num-demos", type=int, default=64)
p.add_argument("--max-attempts", type=int, default=256)
p.add_argument("--max-steps", type=int, default=420)
p.add_argument("--episode-length-s", type=float, default=40.0)
p.add_argument("--contact-offset-m", type=float, default=0.0032)
p.add_argument("--direction-sector", type=int, default=-1, help="-1 cycles 8 sectors; 0..7 fixes one sector for calibration.")
p.add_argument("--contact-history-steps", type=int, default=30)
p.add_argument("--min-contact-torque-delta", type=float, default=0.03)
p.add_argument("--max-grasp-drift-m", type=float, default=0.003)
p.add_argument("--resolution", type=int, default=84)
p.add_argument("--hand-noise-xy-m", type=float, default=0.006)
p.add_argument("--held-noise-xy-m", type=float, default=0.0025)
p.add_argument("--seed", type=int, default=20261090)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.num_demos < 1 or a.contact_offset_m <= 0 or a.contact_history_steps < 30 or a.direction_sector not in range(-1, 8):
    p.error("invalid contact-recovery collection settings")

app = AppLauncher(a).app

import h5py
import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


PHASE = {"lift": 0, "offset_setup": 1, "offset_approach": 2, "contact_history": 3, "unload": 4, "recenter": 5, "insert": 6}


def npv(x): return x.detach().float().cpu().numpy()


def state(e):
    return np.concatenate((npv(e.joint_pos[0, :9]), npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)


def pose(e):
    held, fixed = npv(e.held_pos[0]), npv(e.fixed_pos[0])
    d = held - fixed
    return float(np.linalg.norm(d[:2])), float(d[2]), d.astype(np.float32)


def strict(e):
    xy, z, _ = pose(e)
    return xy < 0.0025 and z < 0.001, xy, z


def grasp_anchor(e): return npv(e.fingertip_midpoint_pos[0]) - npv(e.held_pos[0])


def action_to(e, xy_offset: np.ndarray, z: float, clip: float = 0.35):
    fixed, held = npv(e.fixed_pos[0]), npv(e.held_pos[0])
    target = fixed + np.array([xy_offset[0], xy_offset[1], z], np.float32)
    u = np.zeros(6, np.float32)
    u[:3] = np.clip((target - held) / 0.01, -clip, clip)
    return torch.from_numpy(u).to(e.device)[None]


def cameras(resolution: int):
    result = {}
    for name, pos in {"rgb_table": (1.10, 0.0, 0.80), "rgb_side": (0.65, -0.85, 0.52)}.items():
        camera = rep.create.camera(position=pos, look_at=(0.0, 0.0, 0.30))
        product = rep.create.render_product(camera, resolution=(resolution, resolution))
        product = product if isinstance(product, str) else product.path
        ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        ann.attach(product)
        result[name] = ann
    return result


def image(ann):
    x = np.asarray(ann.get_data())
    x = x[..., :3] if x.shape[-1] == 4 else x
    if x.ndim != 3 or x.shape[-1] != 3: raise RuntimeError(f"invalid RGB {x.shape}; use --enable_cameras")
    return x.astype(np.uint8, copy=False)


cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
cfg.seed = a.seed
cfg.episode_length_s = a.episode_length_s
cfg.task.hand_init_pos_noise = [a.hand_noise_xy_m, a.hand_noise_xy_m, 0.004]
cfg.task.held_asset_pos_noise = [a.held_noise_xy_m, a.held_noise_xy_m, 0.001]
cfg.sim.render_interval = cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
e = env.unwrapped
anns = cameras(a.resolution)
a.dataset_file.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(a.dataset_file, "a") as f:
    f.attrs.update(
        format="factory_peg_insert_contact_recovery_native_v1",
        collection_integrity="native reset + controller-only physical contact/recovery; all observations pre-action",
        state_schema="proprio12:[joint_pos(9),fingertip_midpoint_pos(3)]; no peg/hole truth",
        action_schema="factory_delta_pose6:[xyz(3),axis_angle(3)]",
        audit_schema="pose/contact/phase values are audit-only and never policy inputs",
        policy_contract="is_policy_label is true only for post-contact unload/recenter/insert actions",
    )
    demos = f.require_group("demos")
    saved = len(demos)
    print(f"[CONTACT_RECOVERY_NATIVE] resume={saved}/{a.num_demos} file={a.dataset_file}", flush=True)
    for attempt in range(1, a.max_attempts + 1):
        if saved >= a.num_demos: break
        seed = a.seed + attempt
        env.reset(seed=seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        for _ in range(3):
            env.step(torch.zeros((1, 6), dtype=torch.float32, device=e.device)); e.sim.render()
        sector = a.direction_sector if a.direction_sector >= 0 else (attempt - 1) % 8
        angle = 2 * np.pi * sector / 8
        offset = np.array([np.cos(angle), np.sin(angle)], np.float32) * a.contact_offset_m
        buf = {k: [] for k in ("state", "action", "rgb_table", "rgb_side", "joint_torque", "applied_wrench", "phase", "is_policy_label", "audit_xy_error_m", "audit_depth_m")}
        phase = "lift"; phase_step = 0; torque_history: list[np.ndarray] = []
        initial_anchor = grasp_anchor(e); contact_xy = contact_z = float("nan"); contact_torque_delta = 0.0
        first_recenter_dot: list[float] = []; max_grasp_drift = 0.0; success = False; reject = None

        for step in range(a.max_steps):
            xy, z, delta = pose(e)
            phase_action = phase
            if phase == "lift":
                u = action_to(e, np.zeros(2, np.float32), 0.030)
                if phase_step >= 24: phase, phase_step = "offset_setup", 0
            elif phase == "offset_setup":
                u = action_to(e, offset, 0.030)
                if phase_step >= 23: phase, phase_step = "offset_approach", 0
            elif phase == "offset_approach":
                # Descend physically while remaining off-centre.  The final push
                # below is deliberately not a policy label.
                depth = max(-0.002, 0.025 - 0.00027 * phase_step)
                u = action_to(e, offset, depth)
                if phase_step in (1, 25, 50, 75, 99):
                    print(f"[CONTACT_RECOVERY_NATIVE] approach_trace phase_step={phase_step} xy={xy:.5f} z={z:.5f} target_z={depth:.5f} action_z={float(u[0,2]):.4f}", flush=True)
                if phase_step >= 99: phase, phase_step = "contact_history", 0
            elif phase == "contact_history":
                u = action_to(e, offset, -0.002, clip=0.20)
                if phase_step <= 1:
                    contact_xy, contact_z = xy, z
                    if z < 0.015 or z > 0.032:
                        reject = "contact_height_out_of_band"
                    elif xy < 0.0025 or xy > 0.009:
                        reject = "contact_xy_out_of_band"
                torque_history.append(npv(e.joint_torque[0, :7]))
                if len(torque_history) >= a.contact_history_steps:
                    base = np.asarray(torque_history[: max(5, len(torque_history)//3)])
                    now = np.asarray(torque_history)
                    contact_torque_delta = float(np.max(np.linalg.norm(now - np.median(base, axis=0), axis=1)))
                    already_success, _, _ = strict(e)
                    if already_success: reject = "inserted_before_recovery"
                    elif contact_torque_delta < a.min_contact_torque_delta: reject = "no_contact_torque_excursion"
                    elif max_grasp_drift > a.max_grasp_drift_m: reject = "grasp_drift_contact"
                    else: phase, phase_step = "unload", 0
            elif phase == "unload":
                u = action_to(e, offset, 0.012)
                if phase_step >= 14: phase, phase_step = "recenter", 0
            elif phase == "recenter":
                u = action_to(e, np.zeros(2, np.float32), 0.012)
                centre = -delta[:2]
                if np.linalg.norm(centre) > 1e-8 and np.linalg.norm(u.detach().cpu().numpy()[0, :2]) > 1e-8:
                    first_recenter_dot.append(float(np.dot(u.detach().cpu().numpy()[0, :2], centre) / (np.linalg.norm(u.detach().cpu().numpy()[0, :2]) * np.linalg.norm(centre))))
                if phase_step >= 19: phase, phase_step = "insert", 0
            else:
                u = action_to(e, np.zeros(2, np.float32), -0.006, clip=0.25)

            # Store precisely the observation used to select u.
            policy = phase_action in ("unload", "recenter", "insert")
            buf["state"].append(state(e)); buf["action"].append(npv(u[0]))
            buf["rgb_table"].append(image(anns["rgb_table"])); buf["rgb_side"].append(image(anns["rgb_side"]))
            buf["joint_torque"].append(npv(e.joint_torque[0, :7])); buf["applied_wrench"].append(npv(e.applied_wrench[0]))
            buf["phase"].append(np.array([PHASE[phase_action]], np.uint8)); buf["is_policy_label"].append(np.array([policy], np.uint8))
            buf["audit_xy_error_m"].append(np.array([xy], np.float32)); buf["audit_depth_m"].append(np.array([z], np.float32))
            env.step(u); e.sim.render()
            max_grasp_drift = max(max_grasp_drift, float(np.linalg.norm(grasp_anchor(e) - initial_anchor)))
            phase_step += 1
            if reject: break
            success, final_xy, final_z = strict(e)
            if phase_action == "insert" and success: break

        final_xy, final_z, _ = pose(e)
        policy_n = int(np.asarray(buf["is_policy_label"], bool).sum())
        direction_ok = bool(first_recenter_dot) and float(np.mean(first_recenter_dot)) >= 0.70
        if not success and reject is None: reject = "not_strict_recovered"
        if policy_n < 20 and reject is None: reject = "insufficient_recovery_labels"
        if not direction_ok and reject is None: reject = "bad_recenter_direction"
        if max_grasp_drift > a.max_grasp_drift_m and reject is None: reject = "grasp_drift"
        if reject is not None:
            print(f"[CONTACT_RECOVERY_NATIVE] reject attempt={attempt} reason={reject} phase={phase} contact_xy={contact_xy:.5f} contact_z={contact_z:.5f} torque_delta={contact_torque_delta:.4f} drift={max_grasp_drift:.4f} final_xy={final_xy:.5f} final_z={final_z:.5f}", flush=True)
            continue
        g = demos.create_group(f"demo_{saved:05d}")
        for key, values in buf.items(): g.create_dataset(key, data=np.stack(values), compression="gzip", compression_opts=1, shuffle=True)
        g.attrs.update(strict_success=True, frame_alignment="pre_action", state_intervention=False, attempt=attempt, episode_seed=seed, pair_id=f"seed_{seed:08d}", direction_sector=int(sector), contact_offset_command_m=float(a.contact_offset_m), contact_xy_error_m=float(contact_xy), contact_depth_m=float(contact_z), contact_torque_delta=float(contact_torque_delta), contact_history_frames=int(len(torque_history)), recovery_label_frames=policy_n, max_grasp_drift_m=float(max_grasp_drift), recenter_direction_dot=float(np.mean(first_recenter_dot)), final_xy_error_m=float(final_xy), final_depth_m=float(final_z))
        f.flush(); saved += 1
        print(f"[CONTACT_RECOVERY_NATIVE] saved={saved}/{a.num_demos} attempt={attempt} steps={len(buf['state'])} contact_xy={contact_xy:.5f} contact_z={contact_z:.5f} torque_delta={contact_torque_delta:.4f} labels={policy_n} final_xy={final_xy:.5f} final_z={final_z:.5f}", flush=True)

env.close(); app.close()
if saved < a.num_demos: raise SystemExit(f"incomplete contact recovery dataset: {saved}/{a.num_demos}")
print(f"[CONTACT_RECOVERY_NATIVE] complete strict_demos={saved}", flush=True)
