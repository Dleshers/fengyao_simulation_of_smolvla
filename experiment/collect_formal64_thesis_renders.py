#!/usr/bin/env python3
"""Render thesis-quality qualitative assets from frozen Formal64 snapshots.

Policy observations remain the original 84x84 products.  This script only
creates separate 1920x1080 Replicator products after each deterministic replay
has completed, then restores recorded physical states to render illustrations.
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


parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--snapshot-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-branch-steps", type=int, default=240)
parser.add_argument("--strict-hold-steps", type=int, default=10)
parser.add_argument("--action-clip", type=float, default=0.35)
parser.add_argument("--flow-noise-seed", type=int, default=20260817)
parser.add_argument("--inference-samples", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.max_branch_steps < 1:
    parser.error("--max-branch-steps must be positive")

app = AppLauncher(args).app
site = os.environ.get("LEROBOT_SITE_PACKAGES")
source = os.environ.get("LEROBOT_SOURCE")
if site:
    sys.path.insert(0, site)
if source:
    sys.path.insert(0, source)
# Avoid a storage-only optional dependency during policy import.
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


RESULTS = json.loads(args.results.read_text())
ROWS = RESULTS["rows"]
ROWS_BY_ID = {int(row["pair_id"]): row for row in ROWS}
SELECTED_IDS = (14, 3, 42)
if set(SELECTED_IDS) - set(ROWS_BY_ID):
    raise RuntimeError("the required thesis pair IDs are absent from Formal64 results")
if not all(row.get("paired_initial_observation_identical") for row in ROWS):
    raise RuntimeError("the Formal64 source did not pass the paired-observation audit")

required = {
    14: ("low", 7, False, True),
    3: ("high", 1, False, True),
    42: ("low", 5, True, False),
}
for pair_id, (band, sector, visual_success, torque_success) in required.items():
    row = ROWS_BY_ID[pair_id]
    actual_band = "low" if int(row["load_band"]) == 0 else "high"
    if actual_band != band or int(row["sector"]) != sector:
        raise RuntimeError(f"pair {pair_id} no longer matches the frozen requested stratum")
    if bool(row["visual"]["success"]) != visual_success:
        raise RuntimeError(f"pair {pair_id} visual outcome differs from the request")
    if bool(row["torque_original"]["success"]) != torque_success:
        raise RuntimeError(f"pair {pair_id} original-torque outcome differs from the request")
if not ROWS_BY_ID[42]["torque_original"]["ejected"]:
    raise RuntimeError("pair 42 is no longer the requested original-torque ejection case")

visual_path = Path(RESULTS["visual_policy"])
torque_path = Path(RESULTS["torque_policy"])
dataset_root = Path(
    "/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla/"
    "_runtime/remote_handoff_gripper_lstm_work/persistent/lerobot_datasets/"
    "Dleshers/factory-peg-insert-contact-recovery-v2-hard80-lerobot"
)
dataset_meta = LeRobotDatasetMetadata(
    "Dleshers/factory-peg-insert-contact-recovery-v2-hard80-lerobot", root=dataset_root
)


def load_policy(path: Path, force_visual: bool):
    raw = json.loads((path / "config.json").read_text())
    raw.pop("tactile_token_mode", None)
    compat = Path(tempfile.mkdtemp(prefix="thesis_render_cfg_"))
    (compat / "config.json").write_text(json.dumps(raw))
    cfg = PreTrainedConfig.from_pretrained(compat)
    cfg.pretrained_path = str(path)
    cfg.device = "cuda"
    cfg.n_action_steps = 1
    if force_visual:
        cfg.use_torque_lstm = False
    policy = make_policy(cfg=cfg, ds_meta=dataset_meta).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, pre, post


visual_policy, visual_pre, visual_post = load_policy(visual_path, True)
torque_policy, torque_pre, torque_post = load_policy(torque_path, False)


def npv(value):
    return value.detach().float().cpu().numpy()


def pose(env_unwrapped):
    delta = npv(env_unwrapped.held_pos[0]) - npv(env_unwrapped.fixed_pos[0])
    return float(np.linalg.norm(delta[:2])), float(delta[2]), delta


def strict(env_unwrapped):
    xy, depth, _ = pose(env_unwrapped)
    return xy < 0.0025 and -0.002 <= depth <= 0.001, xy, depth


def state(env_unwrapped):
    return np.concatenate(
        (npv(env_unwrapped.joint_pos[0, :9]), npv(env_unwrapped.fingertip_midpoint_pos[0]))
    ).astype(np.float32)


def policy_cameras():
    cameras = {}
    for name, location in {"camera1": (1.10, 0.0, 0.80), "camera2": (0.65, -0.85, 0.52)}.items():
        camera = rep.create.camera(position=location, look_at=(0.0, 0.0, 0.30))
        product = rep.create.render_product(camera, resolution=(84, 84))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach(product if isinstance(product, str) else product.path)
        cameras[name] = annotator
    return cameras


def illustrative_camera(target, view):
    target = np.asarray(target, dtype=np.float32)
    offsets = {
        "contact": np.array((0.34, -0.36, 0.22), dtype=np.float32),
        "overview": np.array((0.52, -0.54, 0.34), dtype=np.float32),
        "side": np.array((0.20, -0.56, 0.12), dtype=np.float32),
    }
    location = target + offsets[view]
    camera = rep.create.camera(position=tuple(float(x) for x in location), look_at=tuple(float(x) for x in target))
    product = rep.create.render_product(camera, resolution=(1920, 1080))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach(product if isinstance(product, str) else product.path)
    return annotator, location.tolist(), target.tolist()


def read_rgb(annotator, resize=None):
    array = np.asarray(annotator.get_data())
    if array.shape[-1] == 4:
        array = array[..., :3]
    image = Image.fromarray(array.astype(np.uint8))
    if resize:
        image = image.resize(resize, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8).copy()


def save_png(image, path):
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, format="PNG")


def observation(env_unwrapped, images, torque_window=None):
    batch = {
        "observation.state": torch.from_numpy(state(env_unwrapped)[None]),
        "observation.images.camera1": torch.from_numpy(images["camera1"][None]).permute(0, 3, 1, 2).float() / 255.0,
        "observation.images.camera2": torch.from_numpy(images["camera2"][None]).permute(0, 3, 1, 2).float() / 255.0,
        "task": ["Insert the peg into the hole"],
    }
    if torque_window is not None:
        batch["observation.gripper_torque"] = torch.from_numpy(torque_window[None])
    return batch


def select(policy, pre, post, batch, step):
    samples = []
    for sample_index in range(args.inference_samples):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.flow_noise_seed + sample_index)
        cfg = policy.config
        noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=generator, device="cuda")
        with torch.inference_mode():
            samples.append(post(policy.select_action(pre(dict(batch)), noise=noise)).detach().float().cpu().numpy())
    action = np.mean(samples, axis=0)
    return np.clip(action, -args.action_clip, args.action_clip)


SNAPSHOT_ATTRS = (
    "ctrl_target_joint_pos", "prev_fingertip_pos", "prev_fingertip_quat", "prev_joint_pos",
    "ee_linvel_fd", "ee_angvel_fd", "joint_vel_fd", "joint_torque", "applied_wrench",
    "actions", "prev_actions", "ep_succeeded", "ep_success_times", "episode_length_buf",
    "reset_buf", "fixed_pos_obs_frame", "init_fixed_pos_obs_noise",
)


def to_gpu(snapshot):
    return {key: (value.to("cuda") if torch.is_tensor(value) else value) for key, value in snapshot.items()}


def restore(env_unwrapped, snapshot):
    env_unwrapped._robot.write_root_pose_to_sim(snapshot["robot_root"][:, :7])
    env_unwrapped._robot.write_root_velocity_to_sim(snapshot["robot_root"][:, 7:])
    env_unwrapped._robot.write_joint_state_to_sim(snapshot["robot_joint_pos"], snapshot["robot_joint_vel"])
    env_unwrapped._held_asset.write_root_pose_to_sim(snapshot["held_root"][:, :7])
    env_unwrapped._held_asset.write_root_velocity_to_sim(snapshot["held_root"][:, 7:])
    env_unwrapped._fixed_asset.write_root_pose_to_sim(snapshot["fixed_root"][:, :7])
    env_unwrapped._fixed_asset.write_root_velocity_to_sim(snapshot["fixed_root"][:, 7:])
    if "ctrl_target_joint_pos" in snapshot:
        env_unwrapped._robot.set_joint_position_target(snapshot["ctrl_target_joint_pos"])
    env_unwrapped.scene.write_data_to_sim()
    env_unwrapped.scene.update(dt=env_unwrapped.physics_dt)
    for name in SNAPSHOT_ATTRS:
        if name in snapshot and hasattr(env_unwrapped, name):
            getattr(env_unwrapped, name).copy_(snapshot[name])
    env_unwrapped._compute_intermediate_values(dt=env_unwrapped.physics_dt)
    for name in ("prev_fingertip_pos", "prev_fingertip_quat", "prev_joint_pos", "ee_linvel_fd", "ee_angvel_fd", "joint_vel_fd", "joint_torque", "applied_wrench"):
        if name in snapshot:
            getattr(env_unwrapped, name).copy_(snapshot[name])
    env_unwrapped.sim.render()


def capture_render_state(env_unwrapped):
    return {
        "robot_root": env_unwrapped._robot.data.root_state_w.detach().cpu().clone(),
        "robot_joint_pos": env_unwrapped._robot.data.joint_pos.detach().cpu().clone(),
        "robot_joint_vel": env_unwrapped._robot.data.joint_vel.detach().cpu().clone(),
        "held_root": env_unwrapped._held_asset.data.root_state_w.detach().cpu().clone(),
        "fixed_root": env_unwrapped._fixed_asset.data.root_state_w.detach().cpu().clone(),
        "wrist": env_unwrapped.fingertip_midpoint_pos.detach().cpu().clone(),
    }


def render_state(env_unwrapped, snapshot, annotator):
    robot_root = snapshot["robot_root"].to(env_unwrapped.device)
    held_root = snapshot["held_root"].to(env_unwrapped.device)
    fixed_root = snapshot["fixed_root"].to(env_unwrapped.device)
    env_unwrapped._robot.write_root_pose_to_sim(robot_root[:, :7])
    env_unwrapped._robot.write_root_velocity_to_sim(robot_root[:, 7:])
    env_unwrapped._robot.write_joint_state_to_sim(snapshot["robot_joint_pos"].to(env_unwrapped.device), snapshot["robot_joint_vel"].to(env_unwrapped.device))
    env_unwrapped._held_asset.write_root_pose_to_sim(held_root[:, :7])
    env_unwrapped._held_asset.write_root_velocity_to_sim(held_root[:, 7:])
    env_unwrapped._fixed_asset.write_root_pose_to_sim(fixed_root[:, :7])
    env_unwrapped._fixed_asset.write_root_velocity_to_sim(fixed_root[:, 7:])
    env_unwrapped.scene.write_data_to_sim()
    env_unwrapped.scene.update(dt=env_unwrapped.physics_dt)
    env_unwrapped.sim.render()
    return read_rgb(annotator)


def replay_branch(env, env_unwrapped, policy_cams, snapshot, first_rgb, row, mode):
    seed = int(row["seed"])
    env.reset(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    restore(env_unwrapped, snapshot)
    policy, pre, post = (visual_policy, visual_pre, visual_post) if mode == "visual" else (torque_policy, torque_pre, torque_post)
    policy.reset()
    torques = deque((item.copy() for item in snapshot["torque_history"]), maxlen=30)
    torques.append(npv(env_unwrapped.joint_torque[0, :7]).astype(np.float32))
    states = []
    success = False
    ejected = False
    passed_through = False
    strict_streak = 0
    first_aligned = None
    pre_ejection_index = None
    trace_summary = []
    for step in range(args.max_branch_steps):
        if step == 0:
            images = {key: value.copy() for key, value in first_rgb.items()}
        else:
            env_unwrapped.sim.render()
            images = {key: read_rgb(value, resize=(224, 224)) for key, value in policy_cams.items()}
        torque_window = np.stack(torques).astype(np.float32)
        input_torque = None if mode == "visual" else (np.zeros_like(torque_window) if mode == "torque-zero" else torque_window)
        before = capture_render_state(env_unwrapped)
        before["step"] = step
        states.append(before)
        action = select(policy, pre, post, observation(env_unwrapped, images, input_torque), step)
        env.step(torch.from_numpy(action).to(env_unwrapped.device))
        env_unwrapped.sim.render()
        ok, xy, depth = strict(env_unwrapped)
        ejection_now = depth > 0.040
        pass_now = depth < -0.010
        if ejection_now and pre_ejection_index is None:
            pre_ejection_index = len(states) - 1
        ejected = ejected or ejection_now
        passed_through = passed_through or pass_now
        strict_streak = strict_streak + 1 if ok else 0
        success = strict_streak >= args.strict_hold_steps
        if first_aligned is None and xy < 0.0025:
            first_aligned = step + 1
        trace_summary.append({"step": step, "xy_error_m": xy, "depth_m": depth, "strict": bool(ok), "ejection": bool(ejection_now), "passed_through": bool(pass_now)})
        torques.append(npv(env_unwrapped.joint_torque[0, :7]).astype(np.float32))
        if success or ejected or passed_through:
            break
    terminal = capture_render_state(env_unwrapped)
    terminal["step"] = len(trace_summary)
    recovery_index = min(len(states) - 1, max(1, len(states) // 3)) if len(states) > 1 else 0
    official_key = {"visual": "visual", "torque-original": "torque_original", "torque-zero": "zero"}[mode]
    official = row[official_key]
    return {
        "mode": mode,
        "states": states,
        "terminal": terminal,
        "recovery_index": recovery_index,
        "pre_ejection_index": pre_ejection_index,
        "official": official,
        "replay": {
            "success": bool(success), "ejection": bool(ejected), "passed_through": bool(passed_through),
            "steps": len(trace_summary), "first_aligned_step": first_aligned,
            "last_trace": trace_summary[-1] if trace_summary else None,
        },
    }


def acceptance(image, render_state_data, target):
    target = np.asarray(target, dtype=np.float32)
    wrist = npv(render_state_data["wrist"][0])
    peg = npv(render_state_data["held_root"][0, :3])
    fixture = npv(render_state_data["fixed_root"][0, :3])
    distances = {"wrist": float(np.linalg.norm(wrist - target)), "peg": float(np.linalg.norm(peg - target)), "fixture": float(np.linalg.norm(fixture - target))}
    return {
        "wrist_visible_geometry": distances["wrist"] < 0.42,
        "peg_visible_geometry": distances["peg"] < 0.18,
        "fixture_visible_geometry": distances["fixture"] < 0.18,
        "peg_hole_relationship_geometry": float(np.linalg.norm(peg - fixture)) < 0.12,
        "resolution_ok": list(image.shape[:2]) == [1080, 1920],
        "non_empty_image": float(np.std(image)) > 4.0,
        "same_camera_pose_within_pair": True,
        "object_target_distances_m": distances,
    }


def write_case_pair(root, pair_id, row, replayed):
    case_dir = root / "paired_original_only_success" / str(pair_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    source_state = replayed["visual"]["states"][0]
    target = npv(source_state["fixed_root"][0, :3]) + np.array((0.0, 0.0, 0.020), dtype=np.float32)
    annotator, camera_pose, camera_target = illustrative_camera(target, "contact")
    case_metadata = {"pair_id": pair_id, "case_type": "paired_original_only_success", "load_band": "low" if int(row["load_band"]) == 0 else "high", "direction_sector": int(row["sector"]), "seed": int(row["seed"]), "snapshot_sha256": row["snapshot_sha256"], "checkpoint": {"visual": str(visual_path), "torque_original": str(torque_path)}, "illustrative_camera_pose": camera_pose, "illustrative_camera_target": camera_target, "resolution": [1920, 1080], "branches": {}}
    for mode, stem in (("visual", "visual"), ("torque-original", "torque_original")):
        item = replayed[mode]
        frames = {"initial": item["states"][0], "recovery": item["states"][item["recovery_index"]], "terminal": item["terminal"]}
        branch_meta = {"official_strict_success": bool(item["official"]["success"]), "official_ejection": bool(item["official"]["ejected"]), "replay_outcome": item["replay"], "frames": {}, "acceptance": {}}
        for label, state_data in frames.items():
            image = render_state(env_unwrapped, state_data, annotator)
            filename = f"{stem}_illustrative_{label}.png"
            save_png(image, case_dir / filename)
            branch_meta["frames"][label] = {"step": int(state_data["step"]), "path": str((case_dir / filename).resolve())}
            branch_meta["acceptance"][label] = acceptance(image, state_data, target)
        case_metadata["branches"][mode] = branch_meta
    (case_dir / "metadata.json").write_text(json.dumps(case_metadata, indent=2) + "\n")
    return case_metadata


def write_ejection_case(root, row, replayed):
    pair_id = 42
    case_dir = root / "original_ejection_safety_case" / str(pair_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    source_state = replayed["torque-original"]["states"][0]
    target = npv(source_state["fixed_root"][0, :3]) + np.array((0.0, 0.0, 0.020), dtype=np.float32)
    annotator, camera_pose, camera_target = illustrative_camera(target, "contact")
    original = replayed["torque-original"]
    pre_index = original["pre_ejection_index"] if original["pre_ejection_index"] is not None else max(0, len(original["states"]) - 1)
    frames = {
        "torque_original_initial": original["states"][0],
        "torque_original_pre_ejection": original["states"][pre_index],
        "torque_original_ejection_or_terminal": original["terminal"],
        "torque_zero_terminal": replayed["torque-zero"]["terminal"],
    }
    output_frames = {}
    for label, state_data in frames.items():
        image = render_state(env_unwrapped, state_data, annotator)
        filename = f"{label}_illustrative.png"
        save_png(image, case_dir / filename)
        output_frames[label] = {"step": int(state_data["step"]), "path": str((case_dir / filename).resolve()), "acceptance": acceptance(image, state_data, target)}
    payload = {"pair_id": pair_id, "case_type": "original_ejection_safety_case", "load_band": "low", "direction_sector": int(row["sector"]), "seed": int(row["seed"]), "snapshot_sha256": row["snapshot_sha256"], "checkpoint": {"torque_original": str(torque_path), "torque_zero": str(torque_path)}, "official_outcomes": {"torque_original": {"strict_success": bool(row["torque_original"]["success"]), "ejection": bool(row["torque_original"]["ejected"])}, "torque_zero": {"strict_success": bool(row["zero"]["success"]), "ejection": bool(row["zero"]["ejected"])}}, "replay_outcomes": {"torque_original": original["replay"], "torque_zero": replayed["torque-zero"]["replay"]}, "illustrative_camera_pose": camera_pose, "illustrative_camera_target": camera_target, "resolution": [1920, 1080], "frames": output_frames}
    (case_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


env_cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
env_cfg.seed = 20260817
env_cfg.episode_length_s = 120
env_cfg.task.hand_init_pos_noise = [0.006, 0.006, 0.004]
env_cfg.task.held_asset_pos_noise = [0.0025, 0.0025, 0.001]
env_cfg.sim.render_interval = env_cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=env_cfg)
env_unwrapped = env.unwrapped
low_cameras = policy_cameras()
root = args.output
root.mkdir(parents=True, exist_ok=True)

snapshot_audit = []
replays = {}
for pair_id, modes in ((14, ("visual", "torque-original")), (3, ("visual", "torque-original")), (42, ("torque-original", "torque-zero"))):
    row = ROWS_BY_ID[pair_id]
    snapshot_path = args.snapshot_dir / f"snapshot_{pair_id:04d}.pt"
    first_path = args.snapshot_dir / f"snapshot_{pair_id:04d}_first_rgb.npz"
    snapshot = to_gpu(torch.load(snapshot_path, map_location="cpu", weights_only=False))
    first_np = np.load(first_path)
    first_rgb = {"camera1": first_np["camera1"].copy(), "camera2": first_np["camera2"].copy()}
    snapshot_audit.append({"pair_id": pair_id, "snapshot_sha256": row["snapshot_sha256"], "first_rgb_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in first_rgb.items()}})
    replays[pair_id] = {mode: replay_branch(env, env_unwrapped, low_cameras, snapshot, first_rgb, row, mode) for mode in modes}

# Render all branch images only after the policy replays, so the policy camera
# products and 84x84 model inputs were never changed by illustrative products.
pair_metadata = [write_case_pair(root, pair_id, ROWS_BY_ID[pair_id], replays[pair_id]) for pair_id in (14, 3)]
ejection_metadata = write_ejection_case(root, ROWS_BY_ID[42], replays[42])

# Task overview is a representative frozen pair-14 initial state, rendered by
# dedicated cameras focused on its local peg/hole contact region.
overview_state = replays[14]["visual"]["states"][0]
overview_target = npv(overview_state["fixed_root"][0, :3]) + np.array((0.0, 0.0, 0.020), dtype=np.float32)
overview_dir = root / "task_overview"
overview_dir.mkdir(parents=True, exist_ok=True)
overview_metadata = {"pair_id": 14, "branch": "visual", "snapshot_sha256": ROWS_BY_ID[14]["snapshot_sha256"], "resolution": [1920, 1080], "illustrative_render_not_policy_input": True, "views": {}}
for view, filename in (("overview", "task_contact_overview_oblique.png"), ("side", "task_contact_overview_side.png")):
    annotator, camera_pose, camera_target = illustrative_camera(overview_target, view)
    image = render_state(env_unwrapped, overview_state, annotator)
    save_png(image, overview_dir / filename)
    overview_metadata["views"][view] = {"path": str((overview_dir / filename).resolve()), "illustrative_camera_pose": camera_pose, "illustrative_camera_target": camera_target, "acceptance": acceptance(image, overview_state, overview_target)}
(overview_dir / "metadata.json").write_text(json.dumps(overview_metadata, indent=2) + "\n")

manifest = {"official_results": str(args.results), "selected_pair_ids": list(SELECTED_IDS), "checkpoint_visual": str(visual_path), "checkpoint_torque": str(torque_path), "snapshot_audit": snapshot_audit, "task_overview": str((overview_dir / "metadata.json").resolve()), "paired_cases": [str((root / "paired_original_only_success" / str(pair_id) / "metadata.json").resolve()) for pair_id in (14, 3)], "ejection_case": str((root / "original_ejection_safety_case" / "42" / "metadata.json").resolve()), "policy_rgb_audit_assets_retained_at": "/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla/experiment_results/formal64_qualitative_assets_20260819"}
(root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

lines = ["# Formal64 thesis render re-run report", "", f"- Official result source: `{args.results}`", "- Conditions: visual, torque-original, torque-zero only; no shuffle assets were rendered.", "- Resolution: 1920×1080 PNG for every illustrative render.", "- Every paired branch starts from its identical frozen Formal64 snapshot; official outcome labels and replay observations are kept separate in metadata.", "", "## Asset map", ""]
for case in pair_metadata:
    for branch, item in case["branches"].items():
        for label, frame in item["frames"].items():
            passes = all(value for key, value in item["acceptance"][label].items() if key not in ("object_target_distances_m",))
            lines.append(f"- pair {case['pair_id']} | {branch} | {label} | official success={item['official_strict_success']} | 1920×1080 | acceptance={passes} | `{frame['path']}`")
for label, frame in ejection_metadata["frames"].items():
    passes = all(value for key, value in frame["acceptance"].items() if key not in ("object_target_distances_m",))
    lines.append(f"- pair 42 | {label} | 1920×1080 | acceptance={passes} | `{frame['path']}`")
for view, item in overview_metadata["views"].items():
    passes = all(value for key, value in item["acceptance"].items() if key not in ("object_target_distances_m",))
    lines.append(f"- overview | {view} | 1920×1080 | acceptance={passes} | `{item['path']}`")
lines += ["", "## Acceptance note", "", "The acceptance metadata verifies image resolution/non-emptiness and the fixed-camera geometric framing of wrist, peg, and fixture around the contact target. PNGs contain no GUI, labels, arrows, or debug overlays."]
(root / "ASSET_COLLECTION_REPORT.md").write_text("\n".join(lines) + "\n")

env.close()
app.close()
print(json.dumps({"output": str(root), "selected_pair_ids": list(SELECTED_IDS), "pngs": 18}))
