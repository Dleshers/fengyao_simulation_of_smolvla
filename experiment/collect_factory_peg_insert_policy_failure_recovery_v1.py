#!/usr/bin/env python3
"""Collect recovery demonstrations from actual sub-mm 5k-policy failure states.

Every episode starts with a native Factory reset. The peg reaches physical rim
contact through controller actions only. A frozen 5k hybrid rollout uses the
visual arm for coarse alignment and latches to the torque arm inside 3.5 mm.
Only trajectories in which that rollout visits a physically blocked <1 mm
state are retained. A bounded oracle then unloads, recentres above the hole,
inserts, and holds strict success.

Policy rollout frames are retained for chronological torque history but never
used as behavior-cloning labels. Only the recovery oracle actions are labels.
Oracle pose truth and failure labels are audit metadata, never policy inputs.
"""
from __future__ import annotations

import argparse
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
p.add_argument("--dataset-file", type=Path, required=True)
p.add_argument("--visual-policy-path", type=Path, required=True)
p.add_argument("--torque-policy-path", type=Path, required=True)
p.add_argument("--dataset-root", type=Path, required=True)
p.add_argument("--repo-id", required=True)
p.add_argument("--num-demos", type=int, default=16)
p.add_argument("--max-attempts", type=int, default=640)
p.add_argument("--seed", type=int, default=20260814)
p.add_argument("--resolution", type=int, default=84)
p.add_argument("--contact-history-steps", type=int, default=30)
p.add_argument("--pre-policy-unload-steps", type=int, default=14)
p.add_argument("--policy-steps", type=int, default=180)
p.add_argument("--inference-samples", type=int, default=3)
p.add_argument("--coarse-until-xy-m", type=float, default=0.0035)
p.add_argument("--deterministic-flow-noise", action="store_true")
p.add_argument("--flow-noise-seed", type=int, default=20260814)
p.add_argument("--action-clip", type=float, default=0.35)
p.add_argument("--min-contact-torque-delta", type=float, default=0.02)
p.add_argument("--max-grasp-drift-m", type=float, default=0.003)
p.add_argument("--hand-noise-xy-m", type=float, default=0.006)
p.add_argument("--held-noise-xy-m", type=float, default=0.0025)
p.add_argument("--strict-hold-steps", type=int, default=10)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.num_demos < 1 or a.num_demos > 64:
    p.error("--num-demos must be in [1,64]")
if a.contact_history_steps != 30:
    p.error("--contact-history-steps must be 30")
if min(a.max_attempts, a.policy_steps, a.inference_samples, a.strict_hold_steps) < 1:
    p.error("attempt, policy, inference and hold counts must be positive")
if a.resolution < 32 or a.action_clip <= 0:
    p.error("invalid image/action settings")

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
import h5py
import numpy as np
import omni.replicator.core as rep
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


PHASE = {
    "contact_history": 3,
    "policy_rollout": 4,
    "recovery_unload": 5,
    "recovery_recenter": 6,
    "recovery_insert": 7,
    "strict_hold": 8,
}


def npv(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def proprio(e) -> np.ndarray:
    return np.concatenate((npv(e.joint_pos[0, :9]), npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)


def pose(e) -> tuple[float, float, np.ndarray]:
    delta = npv(e.held_pos[0]) - npv(e.fixed_pos[0])
    return float(np.linalg.norm(delta[:2])), float(delta[2]), delta.astype(np.float32)


def strict(e) -> tuple[bool, float, float]:
    xy, z, _ = pose(e)
    return xy < 0.0025 and -0.002 <= z <= 0.001, xy, z


def grasp_anchor(e) -> np.ndarray:
    return npv(e.fingertip_midpoint_pos[0]) - npv(e.held_pos[0])


def action_to(e, offset: np.ndarray, z: float, clip: float = 0.35) -> torch.Tensor:
    target = npv(e.fixed_pos[0]) + np.array([offset[0], offset[1], z], np.float32)
    action = np.zeros(6, np.float32)
    action[:3] = np.clip((target - npv(e.held_pos[0])) / 0.01, -clip, clip)
    return torch.from_numpy(action).to(e.device)[None]


def cameras(resolution: int) -> dict[str, object]:
    result = {}
    for name, position in {"rgb_table": (1.10, 0.0, 0.80), "rgb_side": (0.65, -0.85, 0.52)}.items():
        camera = rep.create.camera(position=position, look_at=(0.0, 0.0, 0.30))
        product = rep.create.render_product(camera, resolution=(resolution, resolution))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach(product if isinstance(product, str) else product.path)
        result[name] = annotator
    return result


def capture_images(annotators: dict[str, object]) -> dict[str, np.ndarray]:
    images = {}
    for name, annotator in annotators.items():
        value = np.asarray(annotator.get_data())
        if value.shape[-1] == 4:
            value = value[..., :3]
        if value.ndim != 3 or value.shape[-1] != 3:
            raise RuntimeError(f"invalid RGB shape {value.shape}; use --enable_cameras")
        images[name] = value.astype(np.uint8, copy=True)
    return images


def resize(image: np.ndarray) -> np.ndarray:
    return np.array(
        Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR),
        dtype=np.uint8,
        copy=True,
    )


def policy_batch(e, images: dict[str, np.ndarray], torque_window: np.ndarray | None) -> dict:
    batch = {
        "observation.state": torch.from_numpy(proprio(e)[None]),
        "observation.images.camera1": torch.from_numpy(resize(images["rgb_table"])[None]).permute(0, 3, 1, 2).float() / 255.0,
        "observation.images.camera2": torch.from_numpy(resize(images["rgb_side"])[None]).permute(0, 3, 1, 2).float() / 255.0,
        "task": ["Insert the peg into the hole"],
    }
    if torque_window is not None:
        batch["observation.gripper_torque"] = torch.from_numpy(torque_window[None])
    return batch


def load_policy(path: Path, metadata, force_visual: bool):
    raw = json.loads((path / "config.json").read_text())
    raw.pop("tactile_token_mode", None)
    compat = Path(tempfile.mkdtemp(prefix="submm_recovery_cfg_"))
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


def select_action(policy, pre, post, batch: dict, rollout_step: int, seed: int) -> np.ndarray:
    actions = []
    for sample_idx in range(a.inference_samples):
        noise = None
        if a.deterministic_flow_noise:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(a.flow_noise_seed + rollout_step * 1009 + sample_idx)
            cfg = policy.config
            noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=generator, device="cuda")
        with torch.inference_mode():
            action = post(policy.select_action(pre(batch), noise=noise)).detach().float().cpu().numpy()[0]
        actions.append(action)
    return np.clip(np.mean(actions, axis=0), -a.action_clip, a.action_clip).astype(np.float32)


def make_plan() -> list[dict]:
    plan = []
    for repeat in range(4):
        for sector in range(8):
            for load_band in range(2):
                plan.append({
                    "xy_band": "0.2-1.0mm",
                    "sector": sector,
                    "load_band": load_band,
                    "repeat": repeat,
                    "source_kind": "policy_failure",
                    "policy_arm": "hybrid_visual_torque",
                    "command_radius_m": 0.006,
                })
    return plan


metadata = LeRobotDatasetMetadata(a.repo_id, root=a.dataset_root)
visual_policy, visual_pre, visual_post = load_policy(a.visual_policy_path, metadata, force_visual=True)
torque_policy, torque_pre, torque_post = load_policy(a.torque_policy_path, metadata, force_visual=False)
policies = {
    "visual": (visual_policy, visual_pre, visual_post, False),
    "torque": (torque_policy, torque_pre, torque_post, True),
}

cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
cfg.seed = a.seed
cfg.episode_length_s = 90.0
cfg.task.hand_init_pos_noise = [a.hand_noise_xy_m, a.hand_noise_xy_m, 0.004]
cfg.task.held_asset_pos_noise = [a.held_noise_xy_m, a.held_noise_xy_m, 0.001]
cfg.sim.render_interval = cfg.decimation
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
e = env.unwrapped
annotators = cameras(a.resolution)
plan = make_plan()[: a.num_demos]
a.dataset_file.parent.mkdir(parents=True, exist_ok=True)


def empty_buffer() -> dict[str, list]:
    return {key: [] for key in (
        "state", "action", "rgb_table", "rgb_side", "joint_torque", "applied_wrench",
        "phase", "is_policy_label", "audit_xy_error_m", "audit_depth_m",
    )}


def record(buffer: dict[str, list], images: dict[str, np.ndarray], action: torch.Tensor, phase: str, label: bool):
    xy, z, _ = pose(e)
    buffer["state"].append(proprio(e))
    buffer["action"].append(npv(action[0]).astype(np.float32))
    buffer["rgb_table"].append(images["rgb_table"])
    buffer["rgb_side"].append(images["rgb_side"])
    buffer["joint_torque"].append(npv(e.joint_torque[0, :7]).astype(np.float32))
    buffer["applied_wrench"].append(npv(e.applied_wrench[0]).astype(np.float32))
    buffer["phase"].append(np.array([PHASE[phase]], np.uint8))
    buffer["is_policy_label"].append(np.array([label], np.uint8))
    buffer["audit_xy_error_m"].append(np.array([xy], np.float32))
    buffer["audit_depth_m"].append(np.array([z], np.float32))


with h5py.File(a.dataset_file, "a") as h5:
    h5.attrs.update(
        format="factory_peg_insert_policy_failure_recovery_v1",
        collection_integrity="native reset; controller-only physical contact, policy rollout, oracle recovery",
        state_schema="proprio12:[joint_pos(9),fingertip_midpoint_pos(3)]; no peg/hole truth",
        action_schema="factory_delta_pose6:[xyz(3),axis_angle(3)]",
        policy_contract="only recovery_unload/recenter/insert/hold frames are behavior-cloning labels",
        target_grid="first 16 are balanced over 8 sectors x 2 load bands; up to 4 repeats (64)",
        coarse_until_xy_m=a.coarse_until_xy_m,
        deterministic_flow_noise=a.deterministic_flow_noise,
        visual_policy=str(a.visual_policy_path),
        torque_policy=str(a.torque_policy_path),
    )
    demos = h5.require_group("demos")
    saved = len(demos)
    if saved > len(plan):
        raise RuntimeError(f"existing demos {saved} exceed requested plan {len(plan)}")
    print(f"[SUBMM_RECOVERY] resume={saved}/{len(plan)} file={a.dataset_file}", flush=True)

    attempts = 0
    while saved < len(plan) and attempts < a.max_attempts:
        cell = plan[saved]
        attempts += 1
        seed = a.seed + attempts - 1
        sector = int(cell["sector"])
        angle = 2.0 * np.pi * sector / 8
        offset = np.array([np.cos(angle), np.sin(angle)], np.float32) * float(cell["command_radius_m"])
        contact_target_z = -0.0020 if int(cell["load_band"]) == 0 else -0.0040
        env.reset(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        visual_policy.reset()
        torque_policy.reset()
        initial_anchor = grasp_anchor(e)
        for _ in range(3):
            env.step(torch.zeros((1, 6), device=e.device))
            e.sim.render()

        # Reproduce the validated native 6 mm contact entry. Sub-mm offsets
        # cannot be commanded directly because the peg simply enters the hole;
        # the policy itself must produce the retained near-hole failure state.
        rim_offset = offset.copy()
        for _ in range(25):
            env.step(action_to(e, np.zeros(2, np.float32), 0.030))
            e.sim.render()
        for _ in range(24):
            env.step(action_to(e, rim_offset, 0.030))
            e.sim.render()
        for step in range(100):
            target_z = max(contact_target_z, 0.025 - 0.00027 * step)
            env.step(action_to(e, rim_offset, target_z))
            e.sim.render()
        hold_offset = rim_offset.copy()

        buffer = empty_buffer()
        torque_history: deque[np.ndarray] = deque(maxlen=30)
        contact_xy, contact_z, _ = pose(e)
        for _ in range(a.contact_history_steps):
            images = capture_images(annotators)
            action = action_to(e, hold_offset, contact_target_z, clip=0.20)
            record(buffer, images, action, "contact_history", False)
            torque_history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
            env.step(action)
            e.sim.render()

        history_array = np.stack(torque_history)
        baseline = np.median(history_array[:10], axis=0)
        torque_delta = float(np.max(np.linalg.norm(history_array - baseline, axis=1)))
        already_success, _, _ = strict(e)
        grasp_drift = float(np.linalg.norm(grasp_anchor(e) - initial_anchor))
        reject = None
        if not (0.0025 <= contact_xy <= 0.0090):
            reject = "native_rim_contact_xy_out_of_band"
        elif not (0.015 <= contact_z <= 0.032):
            reject = "contact_height_out_of_band"
        elif torque_delta < a.min_contact_torque_delta:
            reject = "no_contact_torque_excursion"
        elif already_success:
            reject = "inserted_before_recovery"
        elif grasp_drift > a.max_grasp_drift_m:
            reject = "grasp_drift_contact"

        # Reproduce the phase-5 transition; these fixed-timer frames are
        # history only, never training labels.
        if reject is None:
            for _ in range(a.pre_policy_unload_steps):
                images = capture_images(annotators)
                action = action_to(e, hold_offset, 0.012)
                record(buffer, images, action, "policy_rollout", False)
                torque_history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
                env.step(action)
                e.sim.render()

        failure_reason = None
        policy_steps_run = 0
        policy_max_lateral = 0.0
        policy_sign_flips = 0
        policy_blocked_steps = 0
        previous_xy_action = None
        policy_first_action = None
        policy_min_xy = float("inf")
        if reject is None and cell["source_kind"] == "policy_failure":
            fine_takeover = False
            for policy_step in range(a.policy_steps):
                before_xy, before_z, _ = pose(e)
                if before_xy <= a.coarse_until_xy_m:
                    fine_takeover = True
                arm = "torque" if fine_takeover else "visual"
                policy, pre, post, uses_torque = policies[arm]
                images = capture_images(annotators)
                torque_history.append(npv(e.joint_torque[0, :7]).astype(np.float32))
                window = np.stack(torque_history).astype(np.float32)
                batch = policy_batch(e, images, window if uses_torque else None)
                action_np = select_action(policy, pre, post, batch, policy_step, seed)
                action = torch.from_numpy(action_np).to(e.device)[None]
                if policy_first_action is None:
                    policy_first_action = action_np.tolist()
                    print(
                        "[SUBMM_RECOVERY] policy_start",
                        {"seed": seed, "xy": before_xy, "z": before_z, "action": policy_first_action},
                        flush=True,
                    )
                record(buffer, images, action, "policy_rollout", False)
                env.step(action)
                e.sim.render()
                after_success, after_xy, after_z = strict(e)
                policy_steps_run += 1
                policy_min_xy = min(policy_min_xy, after_xy)
                lateral = float(np.linalg.norm(action_np[:2]))
                policy_max_lateral = max(policy_max_lateral, lateral)
                blocked = bool(action_np[2] < -0.08 and abs(after_z - before_z) < 0.00010)
                if blocked:
                    policy_blocked_steps += 1
                flipped = bool(
                    previous_xy_action is not None
                    and np.linalg.norm(previous_xy_action) > 1.0e-5
                    and lateral > 1.0e-5
                    and float(np.dot(previous_xy_action, action_np[:2])) < 0.0
                )
                if flipped:
                    policy_sign_flips += 1
                previous_xy_action = action_np[:2].copy()
                if after_success:
                    reject = "policy_succeeded_not_failure"
                    break
                if after_xy < 0.001 and policy_step >= 1 and (
                    lateral >= 0.08 or blocked or flipped or after_xy >= before_xy
                ):
                    failure_reason = (
                        "excessive_lateral" if lateral >= 0.08
                        else "downward_blocked" if blocked
                        else "oscillation" if flipped
                        else "no_alignment_improvement"
                    )
                    break
            if reject is None and failure_reason is None:
                reject = "no_policy_failure_trigger"

        failure_xy, failure_z, failure_delta = pose(e)
        if reject is None and failure_xy >= 0.001:
            reject = "failure_state_left_submm_band"
        if reject is None and not (0.015 <= failure_z <= 0.032):
            reject = "failure_depth_not_blocked"
        if reject is None and strict(e)[0]:
            reject = "already_strict_at_oracle_takeover"

        # Oracle recovery begins from the measured policy-visited or controlled
        # state. It first unloads at the current XY, then recentres in free
        # space, then descends. No state is teleported.
        recovery_start = len(buffer["state"])
        if reject is None:
            start_offset = failure_delta[:2].copy()
            for _ in range(16):
                images = capture_images(annotators)
                action = action_to(e, start_offset, 0.030, clip=0.20)
                record(buffer, images, action, "recovery_unload", True)
                env.step(action)
                e.sim.render()
            for _ in range(28):
                images = capture_images(annotators)
                action = action_to(e, np.zeros(2, np.float32), 0.030, clip=0.16)
                record(buffer, images, action, "recovery_recenter", True)
                env.step(action)
                e.sim.render()
            reached = False
            for _ in range(100):
                images = capture_images(annotators)
                action = action_to(e, np.zeros(2, np.float32), -0.006, clip=0.20)
                record(buffer, images, action, "recovery_insert", True)
                env.step(action)
                e.sim.render()
                reached, _, _ = strict(e)
                if reached:
                    break
            if not reached:
                reject = "oracle_not_strict"
            else:
                for _ in range(a.strict_hold_steps):
                    images = capture_images(annotators)
                    action = action_to(e, np.zeros(2, np.float32), -0.006, clip=0.12)
                    record(buffer, images, action, "strict_hold", True)
                    env.step(action)
                    e.sim.render()
                    if not strict(e)[0]:
                        reject = "strict_hold_failed"
                        break

        final_success, final_xy, final_z = strict(e)
        max_grasp_drift = float(np.linalg.norm(grasp_anchor(e) - initial_anchor))
        label_count = int(np.asarray(buffer["is_policy_label"], dtype=bool).sum())
        if reject is None and not final_success:
            reject = "final_not_strict"
        if reject is None and label_count < 45:
            reject = "insufficient_recovery_labels"
        if reject is None and max_grasp_drift > a.max_grasp_drift_m:
            reject = "grasp_drift_recovery"

        if reject is not None:
            print(
                "[SUBMM_RECOVERY] reject",
                {
                    "cell": saved, "attempt": attempts, "seed": seed, "reason": reject,
                    "source": cell["source_kind"], "arm": cell["policy_arm"],
                    "contact_xy": contact_xy, "failure_xy": failure_xy,
                    "contact_z": contact_z, "torque_delta": torque_delta,
                    "policy_min_xy": policy_min_xy,
                    "policy_first_action": policy_first_action,
                    "final_xy": final_xy, "final_z": final_z,
                },
                flush=True,
            )
            continue

        group = demos.create_group(f"demo_{saved:05d}")
        for key, values in buffer.items():
            group.create_dataset(key, data=np.stack(values), compression="gzip", compression_opts=1, shuffle=True)
        group.attrs.update(
            strict_success=True,
            strict_hold_steps=a.strict_hold_steps,
            frame_alignment="pre_action",
            state_intervention=False,
            attempt=attempts,
            episode_seed=seed,
            pair_id=f"submm_seed_{seed:08d}",
            xy_band=cell["xy_band"],
            direction_sector=sector,
            load_band=int(cell["load_band"]),
            repeat=int(cell["repeat"]),
            source_kind=cell["source_kind"],
            policy_arm=cell["policy_arm"],
            command_radius_m=float(cell["command_radius_m"]),
            contact_xy_error_m=contact_xy,
            contact_depth_m=contact_z,
            contact_torque_delta=torque_delta,
            contact_history_frames=a.contact_history_steps,
            pre_policy_unload_steps=a.pre_policy_unload_steps,
            policy_steps=policy_steps_run,
            policy_min_xy_error_m=policy_min_xy,
            policy_first_action=json.dumps(policy_first_action),
            policy_failure_reason=failure_reason,
            policy_max_lateral_action=policy_max_lateral,
            policy_sign_flips=policy_sign_flips,
            policy_blocked_steps=policy_blocked_steps,
            failure_xy_error_m=failure_xy,
            failure_depth_m=failure_z,
            recovery_start_frame=recovery_start,
            recovery_label_frames=label_count,
            max_grasp_drift_m=max_grasp_drift,
            final_xy_error_m=final_xy,
            final_depth_m=final_z,
        )
        h5.flush()
        saved += 1
        print(
            "[SUBMM_RECOVERY] saved",
            {
                "saved": saved, "target": len(plan), "attempt": attempts,
                "band": cell["xy_band"], "sector": sector, "load": cell["load_band"],
                "source": cell["source_kind"], "arm": cell["policy_arm"],
                "failure": failure_reason, "failure_xy": failure_xy,
                "labels": label_count, "final_xy": final_xy, "final_z": final_z,
            },
            flush=True,
        )

if saved < len(plan):
    env.close()
    app.close()
    raise SystemExit(f"incomplete sub-mm recovery dataset: {saved}/{len(plan)} after {attempts} attempts")
print(f"[SUBMM_RECOVERY] complete strict_demos={saved}", flush=True)
env.close()
app.close()
