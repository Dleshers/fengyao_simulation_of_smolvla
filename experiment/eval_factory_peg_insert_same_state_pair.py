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
p.add_argument("--fixed-load-band", type=int, choices=(0, 1), help="Hold all accepted pairs at one predeclared load band; sectors remain balanced.")
p.add_argument(
    "--resume-json",
    type=Path,
    help="Resume accepted rows from a prior partial summary; rows are retained and only missing pairs are simulated.",
)
p.add_argument(
    "--attempt-offset",
    type=int,
    default=0,
    help="Offset added to attempt indices and deterministic seeds when resuming a partial run.",
)
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
p.add_argument(
    "--initialization-mode",
    choices=("coarse-prefix", "policy-failure", "controlled-contact"),
    default="coarse-prefix",
    help="Use the legacy prefix, policy-generated failure, or deterministic native-contact initialization.",
)
p.add_argument("--generator-visual-policy-path", type=Path)
p.add_argument("--generator-torque-policy-path", type=Path)
p.add_argument("--generator-policy-steps", type=int, default=180)
p.add_argument("--generator-inference-samples", type=int, default=1)
p.add_argument("--generator-coarse-until-xy-m", type=float, default=0.0035)
p.add_argument("--failure-xy-max-m", type=float, default=0.001)
p.add_argument("--failure-depth-min-m", type=float, default=0.015)
p.add_argument("--failure-depth-max-m", type=float, default=0.032)
p.add_argument("--max-grasp-drift-m", type=float, default=0.003)
p.add_argument("--strict-hold-steps", type=int, default=10)
p.add_argument("--ejection-z-m", type=float, default=0.040)
p.add_argument("--pass-through-z-m", type=float, default=-0.010)
p.add_argument("--save-traces", action="store_true")
p.add_argument("--modes", default="visual,torque_original", help="Comma-separated arms: visual,torque_original,zero,shuffle.")
p.add_argument("--snapshot-dir", type=Path, help="Directory for reusable complete CPU snapshots and shared first RGB frames.")
p.add_argument("--controlled-contact-max-steps", type=int, default=180, help="Maximum native-contact alignment steps; stop at first valid <1 mm crossing.")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.episodes < 1 or a.max_attempts < 1 or a.max_coarse_steps < 1 or a.max_branch_steps < 1:
    p.error("episode, attempt and step limits must be positive")
if a.attempt_offset < 0:
    p.error("--attempt-offset must be non-negative")
if a.fixed_load_band is not None and a.episodes % 8 != 0:
    p.error("--fixed-load-band requires episodes to be a multiple of 8 for balanced sectors")
if a.inference_samples < 1:
    p.error("--inference-samples must be positive")
if a.coarse_until_xy_m <= 0:
    p.error("--coarse-until-xy-m must be positive")
if a.pre_takeover_unload_steps < 0:
    p.error("--pre-takeover-unload-steps must be non-negative")
if a.action_clip < 0 or a.fine_xy_action_clip < 0:
    p.error("action clips must be non-negative")
if a.initialization_mode == "policy-failure" and (
    a.generator_visual_policy_path is None or a.generator_torque_policy_path is None
):
    p.error("policy-failure initialization requires both generator policy paths")
if a.generator_policy_steps < 2:
    p.error("--generator-policy-steps must be at least 2")
if a.generator_inference_samples < 1:
    p.error("--generator-inference-samples must be positive")
if not (0 < a.failure_xy_max_m <= a.coarse_until_xy_m):
    p.error("failure XY threshold must be positive and no greater than the branch threshold")
if not (0 < a.failure_depth_min_m < a.failure_depth_max_m):
    p.error("invalid failure depth band")
if a.strict_hold_steps < 1 or a.pass_through_z_m >= -0.002 or a.ejection_z_m <= 0.001:
    p.error("invalid strict hold or safety thresholds")
if a.controlled_contact_max_steps < 1:
    p.error("--controlled-contact-max-steps must be positive")
mode_names = tuple(x.strip() for x in a.modes.split(",") if x.strip())
allowed_modes = {"visual", "torque_original", "zero", "shuffle"}
if not mode_names or any(x not in allowed_modes for x in mode_names) or len(set(mode_names)) != len(mode_names):
    p.error("--modes must be a non-empty comma-separated subset of visual,torque_original,zero,shuffle")
if "visual" not in mode_names or "torque_original" not in mode_names:
    p.error("formal comparisons require both visual and torque_original arms")

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


def select_action(policy, pre, post, batch: dict, step: int, sample_count: int | None = None) -> np.ndarray:
    sampled = []
    samples = a.inference_samples if sample_count is None else sample_count
    for sample_idx in range(samples):
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


def _cpu_snapshot_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def save_snapshot_bundle(snapshot: dict, first_images: dict[str, np.ndarray], pair_id: int, output_dir: Path, metadata: dict) -> dict:
    """Persist all captured controller/physics tensors plus the shared RGB frame.

    The tensors are CPU copies so a later Isaac process can load them without the
    original CUDA context.  The RGB npz and manifest are kept alongside the .pt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"snapshot_{pair_id:04d}"
    tensor_payload = {k: _cpu_snapshot_value(v) for k, v in snapshot.items() if k != "sha256"}
    tensor_payload["sha256"] = snapshot["sha256"]
    tensor_path = output_dir / f"{stem}.pt"
    rgb_path = output_dir / f"{stem}_first_rgb.npz"
    torch.save(tensor_payload, tensor_path)
    np.savez_compressed(rgb_path, **{k: np.asarray(v, dtype=np.uint8) for k, v in first_images.items()})
    manifest = {
        **metadata,
        "snapshot_sha256": snapshot["sha256"],
        "tensor_file": str(tensor_path),
        "first_rgb_file": str(rgb_path),
        "snapshot_keys": sorted(tensor_payload),
        "tensor_device_on_disk": "cpu",
    }
    manifest_path = output_dir / f"{stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"tensor_file": str(tensor_path), "first_rgb_file": str(rgb_path), "manifest_file": str(manifest_path)}


metadata = LeRobotDatasetMetadata(a.repo_id, root=a.dataset_root)
visual_policy, visual_pre, visual_post = load_policy(a.visual_policy_path, metadata, force_visual=True)
torque_policy, torque_pre, torque_post = load_policy(a.torque_policy_path, metadata, force_visual=False)
if a.initialization_mode == "policy-failure":
    generator_visual_policy, generator_visual_pre, generator_visual_post = load_policy(
        a.generator_visual_policy_path, metadata, force_visual=True
    )
    generator_torque_policy, generator_torque_pre, generator_torque_post = load_policy(
        a.generator_torque_policy_path, metadata, force_visual=False
    )
else:
    generator_visual_policy = generator_visual_pre = generator_visual_post = None
    generator_torque_policy = generator_torque_pre = generator_torque_post = None

env_cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
env_cfg.seed = a.seed
env_cfg.episode_length_s = 120
env_cfg.task.hand_init_pos_noise = [a.hand_noise_xy_m, a.hand_noise_xy_m, 0.004]
env_cfg.task.held_asset_pos_noise = [a.held_noise_xy_m, a.held_noise_xy_m, 0.001]
env_cfg.sim.render_interval = env_cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=env_cfg)
e = env.unwrapped
annotators = cameras()


def branch(
    name: str,
    policy,
    pre,
    post,
    snapshot: dict,
    first_images: dict[str, np.ndarray],
    pair_seed: int,
) -> dict:
    """Run one causal arm from the exact shared snapshot.

    ``zero`` and ``shuffle`` intentionally use the torque policy while
    transforming only its 30x7 input window.  Their state/RGB start is still
    restored and audited exactly like the visual and original-torque arms.
    """
    restore_error = restore_snapshot(e, snapshot)
    policy.reset()
    base_torques = deque((x.copy() for x in snapshot["torque_history"]), maxlen=30)
    # The converted phase-5 sample includes the current pre-action torque.
    base_torques.append(npv(e.joint_torque[0, :7]).astype(np.float32))
    shuffle_rng = np.random.default_rng(pair_seed + 910_000)

    def input_window(window: np.ndarray, step: int) -> np.ndarray:
        if name == "zero":
            return np.zeros_like(window)
        if name == "shuffle":
            return window[shuffle_rng.permutation(window.shape[0])]
        return window

    initial_base_window = np.stack(base_torques).astype(np.float32)
    initial_window = input_window(initial_base_window, 0)
    first_batch, initial_audit = observation(e, annotators, initial_window, image_override=first_images)
    initial_audit["base_torque_window_sha256"] = hashlib.sha256(initial_base_window.tobytes()).hexdigest()
    initial_audit["torque_input_mode"] = name
    if name == "visual":
        first_batch.pop("observation.gripper_torque")
    initial_xy, initial_z = pose(e)
    branch_anchor = anchor(e).copy()
    action_trace, pose_trace = [], []
    success = False
    strict_streak = 0
    first_aligned_step = None
    first_strict_step = None
    max_grasp_drift = 0.0
    ejected = passed_through = False
    for step in range(a.max_branch_steps):
        if step == 0:
            batch = first_batch
        else:
            e.sim.render()
            base_torques.append(npv(e.joint_torque[0, :7]).astype(np.float32))
            base_window = np.stack(base_torques).astype(np.float32)
            window = input_window(base_window, step)
            batch, _ = observation(e, annotators, window)
            if name == "visual":
                batch.pop("observation.gripper_torque")
        action = select_action(policy, pre, post, batch, step)
        if a.fine_xy_action_clip > 0:
            action[:, :2] = np.clip(action[:, :2], -a.fine_xy_action_clip, a.fine_xy_action_clip)
        action_trace.append(action[0].tolist())
        env.step(torch.from_numpy(action).to(e.device))
        strict_now, xy, z = strict(e)
        delta = npv(e.held_pos[0]) - npv(e.fixed_pos[0])
        pose_trace.append((xy, z, float(delta[0]), float(delta[1])))
        if first_aligned_step is None and xy < 0.0025:
            first_aligned_step = step + 1
        if strict_now and first_strict_step is None:
            first_strict_step = step + 1
        strict_streak = strict_streak + 1 if strict_now else 0
        max_grasp_drift = max(max_grasp_drift, float(np.linalg.norm(anchor(e) - branch_anchor)))
        ejected = ejected or z > a.ejection_z_m
        passed_through = passed_through or z < a.pass_through_z_m
        success = strict_streak >= a.strict_hold_steps
        if success:
            break
    poses = np.asarray(pose_trace)
    result = {
        "name": name,
        "uses_torque": name != "visual",
        "torque_input_mode": name,
        "restore_max_abs_errors": restore_error,
        "initial_audit": initial_audit,
        "initial_xy_error_m": initial_xy,
        "initial_depth_m": initial_z,
        "steps": len(action_trace),
        "success": bool(success),
        "strict_hold_steps_required": a.strict_hold_steps,
        "first_aligned_step": first_aligned_step,
        "first_strict_step": first_strict_step,
        "time_to_success_steps": len(action_trace) if success else None,
        "max_grasp_drift_m": max_grasp_drift,
        "grasp_drift_failure": max_grasp_drift > a.max_grasp_drift_m,
        "ejected": bool(ejected),
        "passed_through": bool(passed_through),
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
if a.resume_json is not None:
    if not a.resume_json.exists():
        p.error(f"resume summary does not exist: {a.resume_json}")
    resume_payload = json.loads(a.resume_json.read_text())
    rows = list(resume_payload.get("rows", []))
    if any(not isinstance(row, dict) for row in rows):
        p.error("resume summary rows must be JSON objects")
    print(f"[SAME_STATE_PAIR] resuming {len(rows)} accepted rows from {a.resume_json}", flush=True)
for local_attempt in range(a.max_attempts):
    if len(rows) >= a.episodes:
        break
    attempt = a.attempt_offset + local_attempt
    seed = a.seed + attempt
    if a.fixed_load_band is not None:
        # Predeclared single-load extension; cycles sectors evenly.
        sector = len(rows) % 8
        load_band = a.fixed_load_band
    else:
        sector = (len(rows) // 2) % 8 if a.initialization_mode in ("policy-failure", "controlled-contact") else attempt % 8
        load_band = len(rows) % 2 if a.initialization_mode in ("policy-failure", "controlled-contact") else attempt % 2
    angle = 2 * np.pi * sector / 8
    offset = np.array([np.cos(angle), np.sin(angle)], np.float32) * a.contact_offset_m
    env.reset(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    visual_policy.reset()
    torque_policy.reset()
    if generator_visual_policy is not None:
        generator_visual_policy.reset()
        generator_torque_policy.reset()
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
            contact_target_z = -0.002 if load_band == 0 else -0.004
            action = action_to(e, offset, max(contact_target_z, 0.025 - 0.00027 * phase_step))
            if phase_step >= 99:
                phase, phase_step = "history", 0
        else:
            contact_target_z = -0.002 if load_band == 0 else -0.004
            action = action_to(e, offset, contact_target_z, clip=0.20)
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
    # phase 5. The hard16 collector uses 14 history-only unload actions.
    unload_steps = 14 if a.initialization_mode == "policy-failure" else a.pre_takeover_unload_steps
    for _ in range(unload_steps):
        history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
        env.step(action_to(e, offset, 0.012))
        e.sim.render()

    # One shared visual prefix removes policy-induced differences before the
    # causal fork. The switch is latched at the first threshold crossing.
    reached_threshold = False
    coarse_trace = []
    failure_reason = None
    if a.initialization_mode == "controlled-contact":
        # Deterministic native-contact initializer for the evaluation gate.
        # Starting from the measured rim contact, move laterally toward the
        # hole while holding a positive 15--32 mm height.  The peg remains
        # constrained by the rim; no simulator pose/state is teleported.
        target_z = 0.023 if load_band == 0 else 0.025
        # Adaptive native alignment: stop on the first valid threshold crossing
        # instead of integrating a fixed 90-step prefix that can overshoot the
        # hole.  All acceptance gates remain unchanged (<1 mm, 15--32 mm,
        # non-strict, and bounded grasp drift).
        for rollout_step in range(a.controlled_contact_max_steps):
            history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
            action = action_to(e, np.zeros(2, np.float32), target_z, clip=0.16)
            env.step(action)
            e.sim.render()
            after_xy, after_z = pose(e)
            coarse_trace.append((after_xy, after_z))
            drift = float(np.linalg.norm(anchor(e) - start_anchor))
            reached_threshold = bool(
                after_xy < a.failure_xy_max_m
                and a.failure_depth_min_m <= after_z <= a.failure_depth_max_m
                and not strict(e)[0]
                and drift <= a.max_grasp_drift_m
            )
            if reached_threshold:
                failure_reason = "controlled_near_hole_contact_first_crossing"
                break
        failure_xy, failure_z = pose(e)
        drift = float(np.linalg.norm(anchor(e) - start_anchor))
        if not reached_threshold:
            failure_reason = "controlled_near_hole_contact_timeout"
        if not reached_threshold:
            print(
                "[SAME_STATE_PAIR] controlled-contact reject",
                {"seed": seed, "sector": sector, "load": load_band, "xy": failure_xy, "z": failure_z, "drift": drift},
                flush=True,
            )
            continue
    elif a.initialization_mode == "policy-failure":
        fine_takeover = False
        previous_xy_action = None
        for rollout_step in range(a.generator_policy_steps):
            before_xy, before_z = pose(e)
            fine_takeover = fine_takeover or before_xy <= a.generator_coarse_until_xy_m
            if fine_takeover:
                policy, pre, post, uses_torque = (
                    generator_torque_policy, generator_torque_pre, generator_torque_post, True
                )
            else:
                policy, pre, post, uses_torque = (
                    generator_visual_policy, generator_visual_pre, generator_visual_post, False
                )
            e.sim.render()
            history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
            window = np.stack(history).astype(np.float32)
            batch, _ = observation(e, annotators, window if uses_torque else None)
            action_np = select_action(policy, pre, post, batch, rollout_step, a.generator_inference_samples)
            # Keep the generator in the measured rim-contact band. Without
            # this guard a policy that has almost found the hole can continue
            # downward and enter it, destroying the intended <1 mm blocked
            # initialization (the previous run ended at z~=0 mm). This is
            # an action-space safety clamp only; state remains native physics.
            if before_z < a.failure_depth_min_m:
                target_z = 0.5 * (a.failure_depth_min_m + a.failure_depth_max_m)
                action_np[0, 2] = max(
                    float(action_np[0, 2]),
                    float(np.clip((target_z - before_z) / 0.01, 0.0, a.action_clip)),
                )
            elif before_z <= a.failure_depth_min_m + 0.004:
                action_np[0, 2] = max(float(action_np[0, 2]), 0.0)
            env.step(torch.from_numpy(action_np).to(e.device))
            after_success, after_xy, after_z = strict(e)
            coarse_trace.append((after_xy, after_z))
            action_vec = action_np[0]
            lateral = float(np.linalg.norm(action_vec[:2]))
            blocked = bool(action_vec[2] < -0.08 and abs(after_z - before_z) < 0.00010)
            flipped = bool(
                previous_xy_action is not None
                and float(np.dot(previous_xy_action, action_vec[:2])) < 0.0
            )
            previous_xy_action = action_vec[:2].copy()
            if after_success:
                break
            if after_xy < a.failure_xy_max_m and rollout_step >= 1 and (
                lateral >= 0.08 or blocked or flipped or after_xy >= before_xy
            ):
                failure_reason = "policy_failure_trigger"
                break
        failure_xy, failure_z = pose(e)
        drift = float(np.linalg.norm(anchor(e) - start_anchor))
        reached_threshold = bool(
            failure_reason and failure_xy < a.failure_xy_max_m
            and a.failure_depth_min_m <= failure_z <= a.failure_depth_max_m
            and not strict(e)[0] and drift <= a.max_grasp_drift_m
        )
        if not reached_threshold:
            print(
                "[SAME_STATE_PAIR] policy-failure reject",
                {"seed": seed, "sector": sector, "load": load_band, "xy": failure_xy, "z": failure_z, "drift": drift},
                flush=True,
            )
            continue
    visual_policy.reset()
    for coarse_step in range(0 if reached_threshold else a.max_coarse_steps):
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
    # the fork observation once and replay that exact RGB to every arm; later
    # frames remain live and branch-specific.
    e.sim.render()
    first_images = {name: image(ann) for name, ann in annotators.items()}
    branch_order = list(mode_names) if len(rows) % 2 == 0 else list(reversed(mode_names))
    policy_specs = {
        "visual": (visual_policy, visual_pre, visual_post),
        "torque_original": (torque_policy, torque_pre, torque_post),
        "zero": (torque_policy, torque_pre, torque_post),
        "shuffle": (torque_policy, torque_pre, torque_post),
    }
    branch_results = {}
    for branch_name in branch_order:
        policy, pre, post = policy_specs[branch_name]
        branch_results[branch_name] = branch(
            branch_name, policy, pre, post, snapshot, first_images, pair_seed=seed
        )

    identity_keys = ("state_sha256", "camera1_sha256", "camera2_sha256")
    identity_values = [branch_results[name]["initial_audit"] for name in mode_names]
    matching_state_rgb = all(
        all(audit.get(key) == identity_values[0].get(key) for key in identity_keys)
        for audit in identity_values
    )
    matching_base_torque = len({
        audit.get("base_torque_window_sha256") for audit in identity_values
    }) == 1
    restore_ok = all(
        result["restore_max_abs_errors"]["max_abs"] <= 1.0e-6
        for result in branch_results.values()
    )
    paired_identical = bool(matching_state_rgb and matching_base_torque and restore_ok)
    snapshot_files = {}
    if a.snapshot_dir is not None:
        snapshot_files = save_snapshot_bundle(
            snapshot,
            first_images,
            len(rows),
            a.snapshot_dir,
            {
                "pair_id": len(rows),
                "seed": seed,
                "sector": sector,
                "load_band": load_band,
                "initialization_mode": a.initialization_mode,
                "modes": list(mode_names),
                "contact_xy_error_m": contact_xy,
                "contact_depth_m": contact_z,
                "contact_torque_delta": torque_delta,
                "coarse_steps": len(coarse_trace),
            },
        )
    row = {
        "pair_id": len(rows),
        "attempt": attempt,
        "seed": seed,
        "sector": sector,
        "load_band": load_band,
        "contact_xy_error_m": contact_xy,
        "contact_depth_m": contact_z,
        "contact_torque_delta": torque_delta,
        "initialization_mode": a.initialization_mode,
        "generator_failure_reason": failure_reason,
        "coarse_steps": len(coarse_trace),
        "coarse_min_xy_error_m": min(x[0] for x in coarse_trace),
        "snapshot_sha256": snapshot["sha256"],
        "snapshot_files": snapshot_files,
        "branch_order": branch_order,
        "paired_state_rgb_identical": matching_state_rgb,
        "paired_base_torque_identical": matching_base_torque,
        "paired_initial_observation_identical": paired_identical,
        "branches": branch_results,
        # Keep stable top-level names for existing visual/original reports.
        "visual": branch_results["visual"],
        "torque_original": branch_results["torque_original"],
    }
    for extra_mode in ("zero", "shuffle"):
        if extra_mode in branch_results:
            row[extra_mode] = branch_results[extra_mode]
    if a.save_traces:
        row["coarse_trace_xy_z"] = coarse_trace
    rows.append(row)
    print("[SAME_STATE_PAIR]", json.dumps(row), flush=True)

valid_pairs = [r for r in rows if r["paired_initial_observation_identical"]]
summary = {
    "benchmark": "same_process_same_snapshot_four_arm_v1" if len(mode_names) == 4 else "same_process_same_snapshot_visual_vs_torque_v1",
    "interpretation": (
        "One native contact initialization and one shared visual prefix are restored in-process for every arm. "
        "All admissible pairs must have identical state/RGB/base-torque hashes and <=1e-6 restoration error. "
        "zero and shuffle transform only the torque policy's 30x7 input window."
    ),
    "visual_policy": str(a.visual_policy_path),
    "torque_policy": str(a.torque_policy_path),
    "modes": list(mode_names),
    "initialization_mode": a.initialization_mode,
    "controlled_contact_max_steps": a.controlled_contact_max_steps,
    "generator_visual_policy": str(a.generator_visual_policy_path) if a.generator_visual_policy_path else None,
    "generator_torque_policy": str(a.generator_torque_policy_path) if a.generator_torque_policy_path else None,
    "episodes_requested": a.episodes,
    "attempts_allowed": a.max_attempts,
    "pairs": len(rows),
    "valid_identical_pairs": len(valid_pairs),
    "strict_recoveries_by_mode": {
        mode: sum(bool(r["branches"][mode]["success"]) for r in valid_pairs)
        for mode in mode_names
    },
    "coarse_until_xy_m": a.coarse_until_xy_m,
    "failure_xy_max_m": a.failure_xy_max_m,
    "failure_depth_band_m": [a.failure_depth_min_m, a.failure_depth_max_m],
    "pre_takeover_unload_steps": a.pre_takeover_unload_steps,
    "inference_samples": a.inference_samples,
    "flow_noise_seed": a.flow_noise_seed,
    "flow_noise_fixed_across_steps": a.flow_noise_fixed_across_steps,
    "common_first_rgb_replayed": True,
    "action_clip": a.action_clip,
    "fine_xy_action_clip": a.fine_xy_action_clip,
    "snapshot_dir": str(a.snapshot_dir) if a.snapshot_dir else None,
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
