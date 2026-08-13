#!/usr/bin/env python3
"""Single-process, same-snapshot visual versus torque insertion evaluation.

A common visual policy first reaches a configured near-hole threshold. The
script then captures the complete Factory simulation/control state and runs
two sequential branches after restoring that exact snapshot: visual-only and
torque-original. Initial RGB, proprioception, torque-history and simulator
hashes are recorded for every branch. This avoids treating two independently
launched Isaac simulations as a paired causal comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import sys
import tempfile
import types
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser()
p.add_argument("--visual-policy-path", type=Path, required=True)
p.add_argument("--torque-policy-path", type=Path, required=True)
p.add_argument("--dataset-root", type=Path, required=True)
p.add_argument("--repo-id", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--episodes", type=int, default=8)
p.add_argument("--max-attempts", type=int, default=64)
p.add_argument("--seed", type=int, default=20261813)
p.add_argument("--max-coarse-steps", type=int, default=240)
p.add_argument("--max-branch-steps", type=int, default=240)
p.add_argument("--coarse-until-xy-m", type=float, default=0.0025)
p.add_argument("--pre-takeover-unload-steps", type=int, default=15)
p.add_argument("--contact-offset-m", type=float, default=0.006)
p.add_argument("--contact-history-steps", type=int, default=30)
p.add_argument("--min-contact-torque-delta", type=float, default=0.03)
p.add_argument("--hand-noise-xy-m", type=float, default=0.006)
p.add_argument("--held-noise-xy-m", type=float, default=0.0025)
p.add_argument("--inference-samples", type=int, default=5)
p.add_argument("--flow-noise-seed", type=int, default=20260813)
p.add_argument("--flow-noise-fixed-across-steps", action="store_true")
p.add_argument("--action-clip", type=float, default=0.35)
p.add_argument("--fine-xy-action-clip", type=float, default=0.05)
p.add_argument("--save-traces", action="store_true")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.episodes < 1 or a.max_attempts < 1 or a.max_coarse_steps < 1 or a.max_branch_steps < 1:
    p.error("episode, attempt and step limits must be positive")
if a.inference_samples < 1:
    p.error("--inference-samples must be positive")
if a.coarse_until_xy_m <= 0:
    p.error("--coarse-until-xy-m must be positive")
if a.pre_takeover_unload_steps < 0:
    p.error("--pre-takeover-unload-steps must be non-negative")
if a.action_clip < 0 or a.fine_xy_action_clip < 0:
    p.error("action clips must be non-negative")

if a.contact_history_steps != 30:
    p.error("--contact-history-steps must be 30 for the trained torque LSTM")
app = AppLauncher(a).app
site = os.environ.get("LEROBOT_SITE_PACKAGES")
source = os.environ.get("LEROBOT_SOURCE")
if site:
    sys.path.insert(0, site)
if source:
    sys.path.insert(0, source)
boto3 = types.ModuleType("boto3")
boto3.__spec__ = importlib.machinery.ModuleSpec("boto3", loader=None)
sys.modules["boto3"] = boto3

import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import torch
from PIL import Image

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


def npv(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def state(e) -> np.ndarray:
    return np.concatenate((npv(e.joint_pos[0, :9]), npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)


def pose(e) -> tuple[float, float]:
    delta = npv(e.held_pos[0]) - npv(e.fixed_pos[0])
    return float(np.linalg.norm(delta[:2])), float(delta[2])


def strict(e) -> tuple[bool, float, float]:
    xy, z = pose(e)
    return xy < 0.0025 and -0.002 <= z <= 0.001, xy, z


def anchor(e) -> np.ndarray:
    return npv(e.fingertip_midpoint_pos[0]) - npv(e.held_pos[0])


def action_to(e, offset: np.ndarray, z: float, clip: float = 0.35) -> torch.Tensor:
    target = npv(e.fixed_pos[0]) + np.array([offset[0], offset[1], z], np.float32)
    action = np.zeros(6, np.float32)
    action[:3] = np.clip((target - npv(e.held_pos[0])) / 0.01, -clip, clip)
    return torch.from_numpy(action).to(e.device)[None]


def cameras() -> dict[str, object]:
    result = {}
    for name, position in {"camera1": (1.10, 0.0, 0.80), "camera2": (0.65, -0.85, 0.52)}.items():
        camera = rep.create.camera(position=position, look_at=(0.0, 0.0, 0.30))
        product = rep.create.render_product(camera, resolution=(84, 84))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach(product if isinstance(product, str) else product.path)
        result[name] = annotator
    return result


def image(annotator) -> np.ndarray:
    value = np.asarray(annotator.get_data())
    if value.shape[-1] == 4:
        value = value[..., :3]
    return np.array(
        Image.fromarray(value.astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR),
        dtype=np.uint8,
        copy=True,
    )


def observation(
    e,
    annotators: dict[str, object],
    torque_window: np.ndarray | None = None,
    image_override: dict[str, np.ndarray] | None = None,
) -> tuple[dict, dict]:
    images = image_override if image_override is not None else {name: image(ann) for name, ann in annotators.items()}
    proprio = state(e)
    batch = {
        "observation.state": torch.from_numpy(proprio[None]),
        "observation.images.camera1": torch.from_numpy(images["camera1"][None]).permute(0, 3, 1, 2).float() / 255.0,
        "observation.images.camera2": torch.from_numpy(images["camera2"][None]).permute(0, 3, 1, 2).float() / 255.0,
        "task": ["Insert the peg into the hole"],
    }
    if torque_window is not None:
        batch["observation.gripper_torque"] = torch.from_numpy(torque_window[None])
    audit = {
        "state_sha256": hashlib.sha256(proprio.tobytes()).hexdigest(),
        "camera1_sha256": hashlib.sha256(images["camera1"].tobytes()).hexdigest(),
        "camera2_sha256": hashlib.sha256(images["camera2"].tobytes()).hexdigest(),
    }
    if torque_window is not None:
        audit["torque_window_sha256"] = hashlib.sha256(torque_window.tobytes()).hexdigest()
    return batch, audit


def load_policy(path: Path, metadata, force_visual: bool):
    raw = json.loads((path / "config.json").read_text())
    raw.pop("tactile_token_mode", None)
    compat = Path(tempfile.mkdtemp(prefix="same_state_pair_cfg_"))
    (compat / "config.json").write_text(json.dumps(raw))
    cfg = PreTrainedConfig.from_pretrained(compat)
    cfg.pretrained_path = str(path)
    cfg.device = "cuda"
    cfg.n_action_steps = 1
    if force_visual:
        cfg.use_torque_lstm = False
    policy = make_policy(cfg=cfg, ds_meta=metadata).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, pre, post


def select_action(policy, pre, post, batch: dict, step: int) -> np.ndarray:
    sampled = []
    for sample_idx in range(a.inference_samples):
        generator = torch.Generator(device="cuda")
        noise_step = 0 if a.flow_noise_fixed_across_steps else step * 1009
        generator.manual_seed(a.flow_noise_seed + sample_idx + noise_step)
        cfg = policy.config
        noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=generator, device="cuda")
        with torch.inference_mode():
            action = post(policy.select_action(pre(dict(batch)), noise=noise)).detach().float().cpu().numpy()
        sampled.append(action)
    action = np.mean(sampled, axis=0)
    if a.action_clip > 0:
        action = np.clip(action, -a.action_clip, a.action_clip)
    return action


SNAPSHOT_ATTRS = (
    "ctrl_target_joint_pos",
    "prev_fingertip_pos",
    "prev_fingertip_quat",
    "prev_joint_pos",
    "ee_linvel_fd",
    "ee_angvel_fd",
    "joint_vel_fd",
    "joint_torque",
    "applied_wrench",
    "actions",
    "prev_actions",
    "ep_succeeded",
    "ep_success_times",
    "episode_length_buf",
    "reset_buf",
    "fixed_pos_obs_frame",
    "init_fixed_pos_obs_noise",
)


def capture_snapshot(e, torque_history: deque[np.ndarray]) -> dict:
    snapshot = {
        "robot_root": e._robot.data.root_state_w.clone(),
        "robot_joint_pos": e._robot.data.joint_pos.clone(),
        "robot_joint_vel": e._robot.data.joint_vel.clone(),
        "held_root": e._held_asset.data.root_state_w.clone(),
        "fixed_root": e._fixed_asset.data.root_state_w.clone(),
        "torque_history": np.stack(torque_history).astype(np.float32),
    }
    for name in SNAPSHOT_ATTRS:
        if hasattr(e, name):
            value = getattr(e, name)
            if torch.is_tensor(value):
                snapshot[name] = value.clone()
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        value = snapshot[name]
        digest.update(name.encode())
        digest.update(npv(value).tobytes() if torch.is_tensor(value) else np.asarray(value).tobytes())
    snapshot["sha256"] = digest.hexdigest()
    return snapshot


def restore_snapshot(e, snapshot: dict) -> dict:
    e._robot.write_root_pose_to_sim(snapshot["robot_root"][:, :7])
    e._robot.write_root_velocity_to_sim(snapshot["robot_root"][:, 7:])
    e._robot.write_joint_state_to_sim(snapshot["robot_joint_pos"], snapshot["robot_joint_vel"])
    e._held_asset.write_root_pose_to_sim(snapshot["held_root"][:, :7])
    e._held_asset.write_root_velocity_to_sim(snapshot["held_root"][:, 7:])
    e._fixed_asset.write_root_pose_to_sim(snapshot["fixed_root"][:, :7])
    e._fixed_asset.write_root_velocity_to_sim(snapshot["fixed_root"][:, 7:])
    if "ctrl_target_joint_pos" in snapshot:
        e._robot.set_joint_position_target(snapshot["ctrl_target_joint_pos"])
    e.scene.write_data_to_sim()
    e.scene.update(dt=e.physics_dt)
    for name in SNAPSHOT_ATTRS:
        if name in snapshot and hasattr(e, name):
            getattr(e, name).copy_(snapshot[name])
    e._compute_intermediate_values(dt=e.physics_dt)
    # _compute_intermediate_values advances finite-difference caches. Restore
    # both their inputs and outputs, plus the current torque sample, so both
    # branches start from the exact same controller and tactile history.
    cache_names = (
        "prev_fingertip_pos", "prev_fingertip_quat", "prev_joint_pos",
        "ee_linvel_fd", "ee_angvel_fd", "joint_vel_fd",
        "joint_torque", "applied_wrench",
    )
    for name in cache_names:
        if name in snapshot:
            getattr(e, name).copy_(snapshot[name])
    e.sim.render()
    errors = {
        "robot_root": float(torch.max(torch.abs(e._robot.data.root_state_w - snapshot["robot_root"])).item()),
        "robot_joint_pos": float(torch.max(torch.abs(e._robot.data.joint_pos - snapshot["robot_joint_pos"])).item()),
        "held_root": float(torch.max(torch.abs(e._held_asset.data.root_state_w - snapshot["held_root"])).item()),
        "fixed_root": float(torch.max(torch.abs(e._fixed_asset.data.root_state_w - snapshot["fixed_root"])).item()),
    }
    errors["max_abs"] = max(errors.values())
    return errors


metadata = LeRobotDatasetMetadata(a.repo_id, root=a.dataset_root)
visual_policy, visual_pre, visual_post = load_policy(a.visual_policy_path, metadata, force_visual=True)
torque_policy, torque_pre, torque_post = load_policy(a.torque_policy_path, metadata, force_visual=False)

env_cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
env_cfg.seed = a.seed
env_cfg.episode_length_s = 120
env_cfg.task.hand_init_pos_noise = [a.hand_noise_xy_m, a.hand_noise_xy_m, 0.004]
env_cfg.task.held_asset_pos_noise = [a.held_noise_xy_m, a.held_noise_xy_m, 0.001]
env_cfg.sim.render_interval = env_cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=env_cfg)
e = env.unwrapped
annotators = cameras()


def branch(name: str, policy, pre, post, snapshot: dict, use_torque: bool, first_images: dict[str, np.ndarray]) -> dict:
    restore_error = restore_snapshot(e, snapshot)
    policy.reset()
    torques = deque((x.copy() for x in snapshot["torque_history"]), maxlen=30)
    # The converted phase-5 sample includes the current pre-action torque.
    # Append it before constructing the first inference window; otherwise the
    # online window is shifted by one frame relative to training.
    torques.append(npv(e.joint_torque[0, :7]).astype(np.float32))
    initial_window = np.stack(torques).astype(np.float32)
    first_batch, initial_audit = observation(e, annotators, initial_window, image_override=first_images)
    if not use_torque:
        first_batch.pop("observation.gripper_torque")
    initial_xy, initial_z = pose(e)
    action_trace, pose_trace = [], []
    success = False
    for step in range(a.max_branch_steps):
        if step == 0:
            batch = first_batch
        else:
            e.sim.render()
            torques.append(npv(e.joint_torque[0, :7]).astype(np.float32))
            window = np.stack(torques).astype(np.float32)
            batch, _ = observation(e, annotators, window)
            if not use_torque:
                batch.pop("observation.gripper_torque")
        action = select_action(policy, pre, post, batch, step)
        if a.fine_xy_action_clip > 0:
            action[:, :2] = np.clip(action[:, :2], -a.fine_xy_action_clip, a.fine_xy_action_clip)
        action_trace.append(action[0].tolist())
        env.step(torch.from_numpy(action).to(e.device))
        success, xy, z = strict(e)
        delta = npv(e.held_pos[0]) - npv(e.fixed_pos[0])
        pose_trace.append((xy, z, float(delta[0]), float(delta[1])))
        if success:
            break
    poses = np.asarray(pose_trace)
    result = {
        "name": name,
        "uses_torque": use_torque,
        "restore_max_abs_errors": restore_error,
        "initial_audit": initial_audit,
        "initial_xy_error_m": initial_xy,
        "initial_depth_m": initial_z,
        "steps": len(action_trace),
        "success": bool(success),
        "min_xy_error_m": float(poses[:, 0].min()),
        "min_depth_m": float(poses[:, 1].min()),
        "final_xy_error_m": float(poses[-1, 0]),
        "final_depth_m": float(poses[-1, 1]),
    }
    if a.save_traces:
        result["action_trace"] = action_trace
        result["pose_trace_xy_z_dx_dy"] = poses.tolist()
    return result


rows = []
for attempt in range(a.max_attempts):
    if len(rows) >= a.episodes:
        break
    seed = a.seed + attempt
    sector = attempt % 8
    angle = 2 * np.pi * sector / 8
    offset = np.array([np.cos(angle), np.sin(angle)], np.float32) * a.contact_offset_m
    env.reset(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    visual_policy.reset()
    torque_policy.reset()
    for _ in range(3):
        env.step(torch.zeros((1, 6), device=e.device))
        e.sim.render()
    phase, phase_step = "lift", 0
    history: deque[np.ndarray] = deque(maxlen=30)
    start_anchor = anchor(e)
    contact_xy = contact_z = torque_delta = 0.0
    valid = False
    for _ in range(260):
        xy, z = pose(e)
        if phase == "lift":
            action = action_to(e, np.zeros(2, np.float32), 0.030)
            if phase_step >= 24:
                phase, phase_step = "offset", 0
        elif phase == "offset":
            action = action_to(e, offset, 0.030)
            if phase_step >= 23:
                phase, phase_step = "approach", 0
        elif phase == "approach":
            action = action_to(e, offset, max(-0.002, 0.025 - 0.00027 * phase_step))
            if phase_step >= 99:
                phase, phase_step = "history", 0
        else:
            action = action_to(e, offset, -0.002, clip=0.20)
            history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
            if phase_step <= 1:
                contact_xy, contact_z = xy, z
            if len(history) >= a.contact_history_steps:
                window = np.stack(history)
                baseline = np.median(window[:10], axis=0)
                torque_delta = float(np.max(np.linalg.norm(window - baseline, axis=1)))
                already_success, _, _ = strict(e)
                drift = float(np.linalg.norm(anchor(e) - start_anchor))
                valid = bool(
                    0.0025 <= contact_xy <= 0.009
                    and 0.015 <= contact_z <= 0.032
                    and torque_delta >= a.min_contact_torque_delta
                    and not already_success
                    and drift <= 0.003
                )
                env.step(action)
                e.sim.render()
                break
        env.step(action)
        e.sim.render()
        phase_step += 1

    if not valid:
        print(
            "[SAME_STATE_PAIR] reject",
            {"attempt": attempt, "seed": seed, "xy": contact_xy, "z": contact_z, "torque_delta": torque_delta},
            flush=True,
        )
        continue

    # Reproduce the collector transition from physical contact history into
    # phase 5. It records pre-action torque for all 15 unload actions.
    for _ in range(a.pre_takeover_unload_steps):
        history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
        env.step(action_to(e, offset, 0.012))
        e.sim.render()

    # One shared visual prefix removes policy-induced differences before the
    # causal fork. The switch is latched at the first threshold crossing.
    reached_threshold = False
    coarse_trace = []
    visual_policy.reset()
    for coarse_step in range(a.max_coarse_steps):
        e.sim.render()
        history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
        batch, _ = observation(e, annotators)
        action_np = select_action(visual_policy, visual_pre, visual_post, batch, coarse_step)
        env.step(torch.from_numpy(action_np).to(e.device))
        xy, z = pose(e)
        coarse_trace.append((xy, z))
        if xy <= a.coarse_until_xy_m:
            reached_threshold = True
            break
    if not reached_threshold:
        print(
            "[SAME_STATE_PAIR] coarse reject",
            {"attempt": attempt, "seed": seed, "min_xy": min(x[0] for x in coarse_trace)},
            flush=True,
        )
        continue

    snapshot = capture_snapshot(e, history)
    # RTX temporal state is not exactly replayable from physics state. Render
    # the fork observation once and replay that exact RGB to both first
    # actions; later frames remain live and branch-specific.
    e.sim.render()
    first_images = {name: image(ann) for name, ann in annotators.items()}
    branch_specs = [
        ("visual", visual_policy, visual_pre, visual_post, False),
        ("torque_original", torque_policy, torque_pre, torque_post, True),
    ]
    # Alternate order to expose any residual branch-order bias.
    if len(rows) % 2:
        branch_specs.reverse()
    branch_results = {}
    for branch_name, policy, pre, post, use_torque in branch_specs:
        branch_results[branch_name] = branch(branch_name, policy, pre, post, snapshot, use_torque, first_images)

    visual = branch_results["visual"]
    torque = branch_results["torque_original"]
    audit_keys = ("state_sha256", "camera1_sha256", "camera2_sha256", "torque_window_sha256")
    matching_hashes = all(visual["initial_audit"].get(k) == torque["initial_audit"].get(k) for k in audit_keys)
    restore_ok = max(
        visual["restore_max_abs_errors"]["max_abs"],
        torque["restore_max_abs_errors"]["max_abs"],
    ) <= 1.0e-6
    paired_identical = bool(matching_hashes and restore_ok)
    row = {
        "pair_id": len(rows),
        "attempt": attempt,
        "seed": seed,
        "sector": sector,
        "contact_xy_error_m": contact_xy,
        "contact_depth_m": contact_z,
        "contact_torque_delta": torque_delta,
        "coarse_steps": len(coarse_trace),
        "coarse_min_xy_error_m": min(x[0] for x in coarse_trace),
        "snapshot_sha256": snapshot["sha256"],
        "branch_order": [x[0] for x in branch_specs],
        "paired_initial_observation_identical": paired_identical,
        "visual": visual,
        "torque_original": torque,
    }
    if a.save_traces:
        row["coarse_trace_xy_z"] = coarse_trace
    rows.append(row)
    print("[SAME_STATE_PAIR]", json.dumps(row), flush=True)

valid_pairs = [r for r in rows if r["paired_initial_observation_identical"]]
summary = {
    "benchmark": "same_process_same_snapshot_visual_vs_torque_v1",
    "interpretation": (
        "One native contact initialization and one shared visual prefix are restored in-process for both branches. "
        "Only pairs with identical state/RGB/torque-window hashes and <=1e-6 restoration error are admissible."
    ),
    "visual_policy": str(a.visual_policy_path),
    "torque_policy": str(a.torque_policy_path),
    "episodes_requested": a.episodes,
    "attempts_allowed": a.max_attempts,
    "pairs": len(rows),
    "valid_identical_pairs": len(valid_pairs),
    "visual_strict_recoveries": sum(r["visual"]["success"] for r in valid_pairs),
    "torque_strict_recoveries": sum(r["torque_original"]["success"] for r in valid_pairs),
    "coarse_until_xy_m": a.coarse_until_xy_m,
    "pre_takeover_unload_steps": a.pre_takeover_unload_steps,
    "inference_samples": a.inference_samples,
    "flow_noise_seed": a.flow_noise_seed,
    "flow_noise_fixed_across_steps": a.flow_noise_fixed_across_steps,
    "common_first_rgb_replayed": True,
    "action_clip": a.action_clip,
    "fine_xy_action_clip": a.fine_xy_action_clip,
    "rows": rows,
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print("[SAME_STATE_PAIR] SUMMARY", json.dumps(summary), flush=True)
if len(rows) < a.episodes:
    env.close()
    raise RuntimeError(f"only produced {len(rows)}/{a.episodes} requested pairs")
if len(valid_pairs) != len(rows):
    env.close()
    raise RuntimeError(f"snapshot identity audit failed for {len(rows) - len(valid_pairs)} pair(s)")
env.close()
app.close()
