#!/usr/bin/env python3
"""Audit temporal phase progression in native-contact SmolVLA action chunks."""
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
p.add_argument("--demos-per-sector", type=int, default=1)
p.add_argument("--phase-min", type=int, default=None, help="Select the first policy-labelled frame at or after this audit phase.")
p.add_argument("--seed", type=int, default=20260813)
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
    return np.asarray(
        Image.fromarray(x.astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
    ).copy()


def history(x: np.ndarray, t: int, n: int = 30) -> np.ndarray:
    y = x[max(0, t - n + 1) : t + 1]
    if len(y) < n:
        y = np.concatenate((np.repeat(y[:1], n - len(y), axis=0), y), axis=0)
    return y.astype(np.float32)


raw = json.loads((a.policy_path / "config.json").read_text())
raw.pop("tactile_token_mode", None)
compat = Path(tempfile.mkdtemp(prefix="native_chunk_cfg_"))
(compat / "config.json").write_text(json.dumps(raw))
cfg = PreTrainedConfig.from_pretrained(compat)
cfg.pretrained_path = str(a.policy_path)
cfg.device = "cuda"
meta = LeRobotDatasetMetadata(a.repo_id, root=a.dataset_root)
policy = make_policy(cfg=cfg, ds_meta=meta).eval()
pre, post = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=str(a.policy_path),
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)

selected: list[tuple[str, int]] = []
counts: dict[int, int] = {}
with h5py.File(a.hdf5, "r") as f:
    for name in sorted(f["demos"]):
        g = f["demos"][name]
        sector = int(g.attrs["direction_sector"])
        if counts.get(sector, 0) >= a.demos_per_sector:
            continue
        labels = np.asarray(g["is_policy_label"]).reshape(-1).astype(bool)
        if a.phase_min is not None and "phase" in g:
            labels &= np.asarray(g["phase"]).reshape(-1) >= a.phase_min
        first = int(np.flatnonzero(labels)[0])
        selected.append((name, first))
        counts[sector] = counts.get(sector, 0) + 1

rows = []
with h5py.File(a.hdf5, "r") as f:
    for i, (name, t) in enumerate(selected):
        g = f["demos"][name]
        common = {
            "observation.state": torch.from_numpy(np.asarray(g["state"][t], np.float32)[None]),
            "observation.images.camera1": torch.from_numpy(resize(np.asarray(g["rgb_table"][t]))[None]).permute(0, 3, 1, 2).float() / 255.0,
            "observation.images.camera2": torch.from_numpy(resize(np.asarray(g["rgb_side"][t]))[None]).permute(0, 3, 1, 2).float() / 255.0,
            "task": ["Insert the peg into the hole"],
        }
        if cfg.use_torque_lstm:
            common["observation.gripper_torque"] = torch.from_numpy(
                history(np.asarray(g["joint_torque"], np.float32), t)[None]
            )
        torch.manual_seed(a.seed + i)
        torch.cuda.manual_seed_all(a.seed + i)
        policy.reset()
        with torch.inference_mode():
            predicted = post(policy.predict_action_chunk(pre(common))).detach().float().cpu().numpy()[0]
        oracle_all = np.asarray(g["action"], np.float32)
        oracle = oracle_all[t : min(t + len(predicted), len(oracle_all))]
        pred = predicted[: len(oracle)]
        dot = np.sum(pred[:, :2] * oracle[:, :2], axis=1)
        denom = np.linalg.norm(pred[:, :2], axis=1) * np.linalg.norm(oracle[:, :2], axis=1)
        cos = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-8)
        rows.append(
            {
                "demo": name,
                "sector": int(g.attrs["direction_sector"]),
                "frame": t,
                "valid_chunk_steps": len(oracle),
                "first_predicted_action": pred[0].tolist(),
                "first_oracle_action": oracle[0].tolist(),
                "first_14_mean_predicted_xy": pred[:14, :2].mean(0).tolist(),
                "steps_14_33_mean_predicted_xy": pred[14:33, :2].mean(0).tolist(),
                "steps_14_33_mean_oracle_xy": oracle[14:33, :2].mean(0).tolist(),
                "xy_cosine_first_14": float(np.mean(cos[:14])),
                "xy_cosine_steps_14_33": float(np.mean(cos[14:33])),
                "predicted_actions": pred.tolist(),
                "oracle_actions": oracle.tolist(),
            }
        )

summary = {
    "policy": str(a.policy_path),
    "samples": len(rows),
    "mean_xy_cosine_first_14": float(np.mean([r["xy_cosine_first_14"] for r in rows])),
    "mean_xy_cosine_steps_14_33": float(np.mean([r["xy_cosine_steps_14_33"] for r in rows])),
    "rows": rows,
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print("[NATIVE_CHUNK_AUDIT]", json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
