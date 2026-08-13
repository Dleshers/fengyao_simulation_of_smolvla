#!/usr/bin/env python3
"""Offline, paired action counterfactual audit for the 7D torque policy.

This does not claim closed-loop success.  It asks the narrower causal question:
with RGB, proprioception, task text and weights held fixed, does replacing a
contact-rich torque history change the predicted corrective action?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--hdf5", type=Path, required=True)
p.add_argument("--policy-path", type=Path, required=True)
p.add_argument("--dataset-root", type=Path, required=True)
p.add_argument("--repo-id", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--samples", type=int, default=24)
p.add_argument("--xy-min-m", type=float, default=None)
p.add_argument("--xy-max-m", type=float, default=None)
p.add_argument(
    "--phase-min",
    type=int,
    default=5,
    help="Only audit policy-labelled frames at or after this recovery phase.",
)
p.add_argument("--seed", type=int, default=88001)
a = p.parse_args()

source = os.environ.get("LEROBOT_SOURCE")
if source:
    sys.path.insert(0, source)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


def resize(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] == 4:
        x = x[..., :3]
    return np.asarray(Image.fromarray(x.astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR))


def history(x: np.ndarray, t: int, n: int = 30) -> np.ndarray:
    y = x[max(0, t - n + 1) : t + 1]
    if len(y) < n:
        y = np.concatenate((np.repeat(y[:1], n - len(y), axis=0), y), axis=0)
    return y.astype(np.float32)


raw = json.loads((a.policy_path / "config.json").read_text())
raw.pop("tactile_token_mode", None)
compat = Path(tempfile.mkdtemp(prefix="torque_cf_cfg_"))
(compat / "config.json").write_text(json.dumps(raw))
cfg = PreTrainedConfig.from_pretrained(compat)
cfg.pretrained_path = str(a.policy_path)
cfg.device = "cuda"
cfg.n_action_steps = 1
meta = LeRobotDatasetMetadata(a.repo_id, root=a.dataset_root)
policy = make_policy(cfg=cfg, ds_meta=meta).eval()
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(a.policy_path), preprocessor_overrides={"device_processor": {"device": "cuda"}})

candidates: list[tuple[str, int, float]] = []
with h5py.File(a.hdf5, "r") as f:
    for name in sorted(f["demos"]):
        g = f["demos"][name]
        tq = np.asarray(g["joint_torque"], np.float32)
        norms = np.linalg.norm(tq, axis=1)
        base = float(np.median(norms[: min(5, len(norms))]))
        labels = (
            np.asarray(g["is_policy_label"]).reshape(-1).astype(bool)
            if "is_policy_label" in g
            else np.ones(len(tq), dtype=bool)
        )
        if "phase" in g:
            labels &= np.asarray(g["phase"]).reshape(-1) >= a.phase_min
        if a.xy_min_m is not None:
            labels &= np.asarray(g["audit_xy_error_m"]).reshape(-1) >= a.xy_min_m
        if a.xy_max_m is not None:
            labels &= np.asarray(g["audit_xy_error_m"]).reshape(-1) < a.xy_max_m
        for t_raw in np.flatnonzero(labels):
            t = int(t_raw)
            candidates.append((name, t, float(norms[t] - base)))
candidates.sort(key=lambda x: x[2], reverse=True)
selected = candidates[: a.samples]
if not selected:
    raise RuntimeError("No recovery/contact-rich frames found")


def infer(batch: dict, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy.reset()
    with torch.inference_mode():
        return post(policy.select_action(pre(batch))).detach().float().cpu().numpy()[0]


rows = []
with h5py.File(a.hdf5, "r") as f:
    for i, (name, t, excursion) in enumerate(selected):
        g = f["demos"][name]
        tq = np.asarray(g["joint_torque"], np.float32)
        win = history(tq, t)
        common = {
            "observation.state": torch.from_numpy(np.asarray(g["state"][t], np.float32)[None]),
            "observation.images.camera1": torch.from_numpy(resize(np.asarray(g["rgb_table"][t]))[None]).permute(0, 3, 1, 2).float() / 255.0,
            "observation.images.camera2": torch.from_numpy(resize(np.asarray(g["rgb_side"][t]))[None]).permute(0, 3, 1, 2).float() / 255.0,
            "task": ["Insert the peg into the hole"],
        }
        actions = {}
        for mode, torque in {
            "original": win,
            "zero": np.zeros_like(win),
            "causal_shuffle": win[np.random.default_rng(a.seed + i).permutation(len(win))],
        }.items():
            batch = dict(common)
            batch["observation.gripper_torque"] = torch.from_numpy(torque[None])
            actions[mode] = infer(batch, a.seed + i)
        oracle = np.asarray(g["action"][t], np.float32)
        rows.append({
            "demo": name, "frame": t, "xy_error_m": float(np.asarray(g["audit_xy_error_m"][t]).reshape(-1)[0]), "torque_excursion_norm": excursion,
            "oracle_action": oracle.tolist(),
            "actions": {k: v.tolist() for k, v in actions.items()},
            "delta_original_zero_l2": float(np.linalg.norm(actions["original"] - actions["zero"])),
            "delta_original_shuffle_l2": float(np.linalg.norm(actions["original"] - actions["causal_shuffle"])),
            "xy_dot_original_oracle": float(np.dot(actions["original"][:2], oracle[:2])),
            "xy_dot_zero_oracle": float(np.dot(actions["zero"][:2], oracle[:2])),
        })
summary = {
    "purpose": "paired offline torque-input counterfactual; not a closed-loop success claim",
    "samples": len(rows),
    "mean_delta_original_zero_l2": float(np.mean([r["delta_original_zero_l2"] for r in rows])),
    "mean_delta_original_shuffle_l2": float(np.mean([r["delta_original_shuffle_l2"] for r in rows])),
    "mean_xy_dot_original_oracle": float(np.mean([r["xy_dot_original_oracle"] for r in rows])),
    "mean_xy_dot_zero_oracle": float(np.mean([r["xy_dot_zero_oracle"] for r in rows])),
    "rows": rows,
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print("[TORQUE_COUNTERFACTUAL_AUDIT]", json.dumps(summary), flush=True)
