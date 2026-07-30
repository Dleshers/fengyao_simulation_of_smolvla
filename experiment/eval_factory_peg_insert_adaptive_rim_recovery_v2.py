#!/usr/bin/env python3
"""Evaluate closed-loop recovery after a controlled near-rim lateral perturbation.

Each episode begins with the grasped peg teleported to a reproducible pose above
the hole: its root is ``initial_depth`` above the hole root and its XY position
is displaced by a random vector whose magnitude is in the requested range.
The Franka hand is first moved by the Factory environment IK routine so this is
still a physically plausible grasp configuration.  The peg is then set exactly
to the perturbation pose, velocities are cleared, and the learned policy must
recover and satisfy the usual strict insertion criterion.

This deliberately measures recovery *after* a near-hole error.  It does not
claim that joint torque is a calibrated contact-force measurement; the saved
torque norm is an auditable proxy for controller/contact loading only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser()
p.add_argument("--policy-path", type=Path, required=True)
p.add_argument("--dataset-root", type=Path, required=True)
p.add_argument("--repo-id", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--seed", type=int, default=5100)
p.add_argument("--max-steps", type=int, default=240)
p.add_argument("--torque-mode", choices=("none", "original", "zero", "shuffle"), default="none")
p.add_argument("--n-action-steps", type=int, default=1)
p.add_argument("--perturb-min", type=float, default=0.002)
p.add_argument("--perturb-max", type=float, default=0.004)
p.add_argument("--initial-depth", type=float, default=0.003)
p.add_argument("--prepare-steps", type=int, default=64, help="Oracle pre-contact steps before kick.")
p.add_argument("--kick-amplitude", type=float, default=0.72, help="Normalized lateral action magnitude.")
p.add_argument("--prepare-max-steps", type=int, default=180, help="Maximum oracle steps used to find the pre-kick near-rim band.")
p.add_argument("--pre-kick-depth-min", type=float, default=0.001)
p.add_argument("--pre-kick-depth-max", type=float, default=0.005)
p.add_argument("--pre-kick-xy-max", type=float, default=0.0025)
p.add_argument("--post-kick-depth-max", type=float, default=0.008)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not (0.0 < a.perturb_min <= a.perturb_max):
    p.error("Require 0 < --perturb-min <= --perturb-max")
if a.initial_depth <= 0.001:
    p.error("--initial-depth must start outside the strict success zone (> 0.001m)")
app = AppLauncher(a).app

site = os.environ.get("LEROBOT_SITE_PACKAGES")
source = os.environ.get("LEROBOT_SOURCE")
if site:
    sys.path.insert(0, site)
if source:
    sys.path.insert(0, source)
for key in list(sys.modules):
    if key == "botocore" or key.startswith("botocore."):
        del sys.modules[key]
import importlib.machinery
import types

_boto3 = types.ModuleType("boto3")
_boto3.__spec__ = importlib.machinery.ModuleSpec("boto3", loader=None)
sys.modules["boto3"] = _boto3

import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


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


def image(annotator):
    data = np.asarray(annotator.get_data())
    data = data[..., :3] if data.shape[-1] == 4 else data
    return np.array(
        Image.fromarray(data.astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR),
        dtype=np.uint8,
        copy=True,
    )


def cameras():
    out = {}
    for name, pos in {"camera1": (1.10, 0.0, 0.80), "camera2": (0.65, -0.85, 0.52)}.items():
        camera = rep.create.camera(position=pos, look_at=(0, 0, 0.30))
        product = rep.create.render_product(camera, resolution=(84, 84))
        product = product if isinstance(product, str) else product.path
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach(product)
        out[name] = annotator
    return out


def set_perturbed_near_rim_state(e, radius, angle):
    """Move the hand near the target, then place the grasped peg exactly there."""
    fixed_pose = e._fixed_asset.data.root_pose_w.clone()
    held_pose_before = e._held_asset.data.root_pose_w.clone()
    hand_pos = e.fingertip_midpoint_pos.clone()
    hand_quat = e.fingertip_midpoint_quat.clone()
    # Preserve the learned/reset grasp transform when moving the robot near the rim.
    hand_minus_held = hand_pos - held_pose_before[:, :3]
    target_held = fixed_pose.clone()
    target_held[:, 0] += float(radius * np.cos(angle))
    target_held[:, 1] += float(radius * np.sin(angle))
    target_held[:, 2] += a.initial_depth
    target_held[:, 3:7] = held_pose_before[:, 3:7]
    target_hand = target_held[:, :3] + hand_minus_held
    env_ids = torch.arange(e.num_envs, device=e.device, dtype=torch.long)
    e.set_pos_inverse_kinematics(target_hand, hand_quat, env_ids)
    zeros = torch.zeros((e.num_envs, 6), device=e.device)
    e._held_asset.write_root_pose_to_sim(target_held[:, :7])
    e._held_asset.write_root_velocity_to_sim(zeros)
    e._held_asset.reset()
    e.step_sim_no_action()
    # Refresh derived tensors, which policy observations access immediately.
    e._compute_intermediate_values(dt=e.physics_dt)
    return pose_error(e)


def oracle_action(e, step):
    """Same pre-contact controller as collect_factory_peg_insert_causal_recovery.py."""
    fixed, held = npv(e.fixed_pos[0]), npv(e.held_pos[0])
    if step < 35:
        target = fixed + np.array([0.0, 0.0, 0.030], np.float32)
    else:
        k = step - 35
        radius = min(0.00035 + 0.000035 * k, 0.0045)
        theta = 0.24 * k
        target = fixed + np.array([radius * np.cos(theta), radius * np.sin(theta), -0.006], np.float32)
    action = np.zeros(6, np.float32)
    action[:3] = np.clip((target - held) / 0.01, -1.0, 1.0)
    return torch.from_numpy(action).to(e.device)[None]


def set_oracle_kick_state(env, e, angle):
    """Find a real near-rim state, then reproduce the collector lateral kick."""
    pre_xy = pre_z = None
    precontact_found = False
    prepare_step = -1
    for prepare_step in range(a.prepare_max_steps):
        env.step(oracle_action(e, prepare_step))
        e.sim.render()
        pre_xy, pre_z = pose_error(e)
        if pre_xy <= a.pre_kick_xy_max and a.pre_kick_depth_min < pre_z <= a.pre_kick_depth_max:
            precontact_found = True
            break
    if not precontact_found:
        return False, prepare_step + 1, pre_xy, pre_z, pre_xy, pre_z
    # Do not inherit the oracle approachs downward EMA action into a lateral kick.
    e.actions.zero_()
    kick = torch.zeros((1, 6), dtype=torch.float32, device=e.device)
    kick[0, 0] = float(a.kick_amplitude * np.cos(angle))
    kick[0, 1] = float(a.kick_amplitude * np.sin(angle))
    env.step(kick)
    e.sim.render()
    initial_xy, initial_z = pose_error(e)
    return True, prepare_step + 1, pre_xy, pre_z, initial_xy, initial_z


raw_cfg = json.loads((a.policy_path / "config.json").read_text())
raw_cfg.pop("tactile_token_mode", None)
compat_dir = Path(tempfile.mkdtemp(prefix="factory_perturb_recovery_cfg_"))
(compat_dir / "config.json").write_text(json.dumps(raw_cfg))
cfg = PreTrainedConfig.from_pretrained(compat_dir)
cfg.pretrained_path = str(a.policy_path)
cfg.device = "cuda"
cfg.n_action_steps = a.n_action_steps
if a.torque_mode == "none":
    cfg.use_torque_lstm = False
meta = LeRobotDatasetMetadata(a.repo_id, root=a.dataset_root)
policy = make_policy(cfg=cfg, ds_meta=meta)
policy.eval()
pre, post = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=str(a.policy_path), preprocessor_overrides={"device_processor": {"device": "cuda"}}
)
env_cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
env_cfg.sim.render_interval = env_cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=env_cfg)
e = env.unwrapped
annotators = cameras()
a.output.parent.mkdir(parents=True, exist_ok=True)

case_rng = np.random.default_rng(a.seed)
shuffle_rng = np.random.default_rng(a.seed + 99991)
rows = []
for ep in range(a.episodes):
    episode_seed = a.seed + ep
    radius = float(case_rng.uniform(a.perturb_min, a.perturb_max))
    angle = float(case_rng.uniform(-np.pi, np.pi))
    env.reset(seed=episode_seed)
    torch.manual_seed(episode_seed)
    torch.cuda.manual_seed_all(episode_seed)
    policy.reset()
    precontact_found, prepare_steps_used, pre_kick_xy, pre_kick_z, initial_xy, initial_z = set_oracle_kick_state(env, e, angle)
    torques = deque(maxlen=30)
    torque_trace = []
    hit = False
    xy, z = initial_xy, initial_z
    first_aligned = None
    for step in range(a.max_steps):
        e.sim.render()
        torque = np.array([float(np.linalg.norm(npv(e.joint_torque[0, :7])))], np.float32)
        torque_trace.append(float(torque[0]))
        torques.append(torque)
        while len(torques) < 30:
            torques.appendleft(torque.copy())
        batch = {
            "observation.state": torch.from_numpy(state(e)[None]),
            "observation.images.camera1": torch.from_numpy(image(annotators["camera1"])[None]).permute(0, 3, 1, 2).float() / 255.0,
            "observation.images.camera2": torch.from_numpy(image(annotators["camera2"])[None]).permute(0, 3, 1, 2).float() / 255.0,
            "task": ["Insert the peg into the hole"],
        }
        if a.torque_mode != "none":
            torque_window = np.stack(torques)
            if a.torque_mode == "zero":
                torque_window = np.zeros_like(torque_window)
            elif a.torque_mode == "shuffle":
                torque_window = torque_window[shuffle_rng.permutation(len(torque_window))]
            batch["observation.gripper_torque"] = torch.from_numpy(torque_window[None])
        with torch.inference_mode():
            action = post(policy.select_action(pre(batch))).detach().float().cpu().numpy()
        env.step(torch.from_numpy(action).to(e.device))
        hit, xy, z = strict(e)
        if first_aligned is None and xy < 0.0025:
            first_aligned = step + 1
        if hit:
            break
    baseline_n = min(5, len(torque_trace))
    baseline = float(np.median(torque_trace[:baseline_n])) if baseline_n else None
    row = {
        "episode": ep,
        "seed": episode_seed,
        "perturb_radius_m_requested": radius,
        "perturb_angle_rad": angle,
        "initialization_mode": "oracle_precontact_then_lateral_kick",
        "prepare_steps": prepare_steps_used,
        "precontact_found": precontact_found,
        "kick_amplitude": a.kick_amplitude,
        "pre_kick_xy_error_m": pre_kick_xy,
        "pre_kick_depth_m": pre_kick_z,
        "initial_depth_requested_m": None,
        "initial_xy_error_m": initial_xy,
        "initial_depth_m": initial_z,
        "valid_non_success_initialization": bool(precontact_found and initial_xy >= 0.0025 and a.pre_kick_depth_min < initial_z <= a.post_kick_depth_max),
        "success": bool(hit),
        "steps": step + 1,
        "first_aligned_step": first_aligned,
        "final_xy_error_m": xy,
        "final_depth_m": z,
        "torque_norm_baseline": baseline,
        "torque_norm_max": float(max(torque_trace)) if torque_trace else None,
        "torque_norm_excursion": float(max(torque_trace) - baseline) if torque_trace and baseline is not None else None,
    }
    rows.append(row)
    print("[PERTURB_RECOVERY_EVAL]", row, flush=True)

valid = [r for r in rows if r["valid_non_success_initialization"]]
aligned = [r for r in valid if r["first_aligned_step"] is not None]
successful = [r for r in valid if r["success"]]
summary = {
    "benchmark": "adaptive_near_rim_lateral_kick_recovery_v2_ema_cleared",
    "important_interpretation": "Initial lateral displacement is a controlled geometric perturbation. torque_norm is a joint-torque proxy, not a calibrated contact force.",
    "torque_mode": a.torque_mode,
    "n_action_steps": cfg.n_action_steps,
    "policy": str(a.policy_path),
    "episodes": a.episodes,
    "perturbation_definition": f"same oracle controller as causal-recovery-v2 until xy<={a.pre_kick_xy_max}m and depth in ({a.pre_kick_depth_min},{a.pre_kick_depth_max}]m (max {a.prepare_max_steps} steps), then one unlabelled XY action kick of magnitude {a.kick_amplitude}; angle uniform [-pi, pi]",
    "strict_definition": "xy<0.0025m and held_z-hole_z<0.001m",
    "valid_initializations": len(valid),
    "alignment_recoveries": len(aligned),
    "alignment_recovery_rate": len(aligned) / len(valid) if valid else None,
    "strict_recoveries": len(successful),
    "strict_recovery_rate": len(successful) / len(valid) if valid else None,
    "mean_steps_to_alignment": float(np.mean([r["first_aligned_step"] for r in aligned])) if aligned else None,
    "mean_steps_to_strict_success": float(np.mean([r["steps"] for r in successful])) if successful else None,
    "rows": rows,
}
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print("[PERTURB_RECOVERY_EVAL] SUMMARY", json.dumps(summary), flush=True)
env.close()
app.close()
