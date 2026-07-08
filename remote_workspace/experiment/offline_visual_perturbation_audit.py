#!/usr/bin/env python3
"""Offline visual perturbation audit for SmolVLA baseline.

This script does not run Isaac and does not train. It loads dataset samples,
perturbs camera images, runs the baseline policy, and measures action drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


TRAIN_STATS = {
    "camera1": {"mean": 0.7332273721694946, "std": 0.2509204149246216},
    "camera2": {"mean": 0.441604882478714, "std": 0.20611362159252167},
}

EVAL_STATS = {
    "camera1": {"mean": 0.72055, "std": 0.25850},
    "camera2": {"mean": 0.59605, "std": 0.26955},
}


def affine_match(img: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    cur_mean = img.mean()
    cur_std = img.std(unbiased=False).clamp_min(1e-6)
    return ((img - cur_mean) / cur_std * std + mean).clamp(0.0, 1.0)


def mean_shift(img: torch.Tensor, mean: float) -> torch.Tensor:
    return (img + (mean - float(img.mean()))).clamp(0.0, 1.0)


def make_variant(sample: dict, variant: str) -> dict:
    obs = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            obs[key] = value.clone()
        else:
            obs[key] = value

    c1 = "observation.images.camera1"
    c2 = "observation.images.camera2"
    if variant == "orig":
        return obs
    if variant == "cam2_eval_affine":
        obs[c2] = affine_match(obs[c2], EVAL_STATS["camera2"]["mean"], EVAL_STATS["camera2"]["std"])
    elif variant == "cam2_eval_mean":
        obs[c2] = mean_shift(obs[c2], EVAL_STATS["camera2"]["mean"])
    elif variant == "cam2_zero":
        obs[c2] = torch.zeros_like(obs[c2])
    elif variant == "cam2_flat_train_mean":
        obs[c2] = torch.full_like(obs[c2], TRAIN_STATS["camera2"]["mean"])
    elif variant == "cam2_flat_eval_mean":
        obs[c2] = torch.full_like(obs[c2], EVAL_STATS["camera2"]["mean"])
    elif variant == "cam1_eval_affine":
        obs[c1] = affine_match(obs[c1], EVAL_STATS["camera1"]["mean"], EVAL_STATS["camera1"]["std"])
    elif variant == "both_eval_affine":
        obs[c1] = affine_match(obs[c1], EVAL_STATS["camera1"]["mean"], EVAL_STATS["camera1"]["std"])
        obs[c2] = affine_match(obs[c2], EVAL_STATS["camera2"]["mean"], EVAL_STATS["camera2"]["std"])
    else:
        raise ValueError(f"unknown variant: {variant}")
    return obs


def select_indices(samples_csv: Path, sample_count: int) -> list[int]:
    rows = []
    with samples_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] == "mid_50pct":
                rows.append(int(row["global_index"]))
    if sample_count >= len(rows):
        return rows
    if sample_count <= 1:
        return rows[:sample_count]
    return [rows[round(i * (len(rows) - 1) / (sample_count - 1))] for i in range(sample_count)]


def action_metrics(pred: torch.Tensor, ref: torch.Tensor, target: torch.Tensor) -> dict:
    pred = pred.float().cpu()
    ref = ref.float().cpu()
    target = target.float().cpu()
    drift = pred - ref
    target_err = pred - target
    return {
        "arm_drift_l2": float(torch.linalg.vector_norm(drift[:7])),
        "arm_drift_linf": float(drift[:7].abs().max()),
        "gripper_drift": float(drift[7]),
        "gripper_pred": float(pred[7]),
        "gripper_sign": -1 if float(pred[7]) < 0 else 1,
        "target_arm_l2": float(torch.linalg.vector_norm(target_err[:7])),
        "target_gripper_abs": float(abs(target_err[7])),
    }


def summarize(rows: list[dict], variants: list[str]) -> dict:
    out = {}
    for variant in variants:
        sub = [r for r in rows if r["variant"] == variant]
        if not sub:
            continue
        out[variant] = {}
        for key in [
            "arm_drift_l2",
            "arm_drift_linf",
            "gripper_drift_abs",
            "target_arm_l2",
            "target_gripper_abs",
        ]:
            vals = sorted(float(r[key]) for r in sub)
            out[variant][key] = {
                "mean": sum(vals) / len(vals),
                "p50": vals[len(vals) // 2],
                "p90": vals[round((len(vals) - 1) * 0.9)],
                "max": max(vals),
            }
        out[variant]["gripper_sign_flip_count_vs_orig"] = sum(
            int(r["gripper_sign_flip_vs_orig"]) for r in sub
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds_meta = LeRobotDatasetMetadata("franka_pickplace_joint_visual_torque_w30_v1", root=args.dataset_root)
    ds = LeRobotDataset("franka_pickplace_joint_visual_torque_w30_v1", root=args.dataset_root)

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = args.device
    cfg.load_vlm_weights = False
    cfg.n_action_steps = 1
    cfg.use_torque_lstm = False

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )

    variants = [
        "orig",
        "cam2_eval_affine",
        "cam2_eval_mean",
        "cam2_zero",
        "cam2_flat_train_mean",
        "cam2_flat_eval_mean",
        "cam1_eval_affine",
        "both_eval_affine",
    ]

    indices = select_indices(args.samples_csv, args.sample_count)
    rows: list[dict] = []
    with torch.no_grad():
        for global_index in indices:
            sample = ds[int(global_index)]
            target = sample["action"].float()
            predictions = {}
            for variant in variants:
                obs = make_variant(sample, variant)
                policy.reset()
                processed = preprocessor(obs)
                action_raw = policy.select_action(processed)
                action = postprocessor(action_raw).squeeze(0).cpu()
                predictions[variant] = action

            ref = predictions["orig"]
            ref_sign = -1 if float(ref[7]) < 0 else 1
            for variant in variants:
                action = predictions[variant]
                metrics = action_metrics(action, ref, target)
                row = {
                    "global_index": int(global_index),
                    "episode_index": int(sample["episode_index"]),
                    "frame_index": int(sample["frame_index"]),
                    "variant": variant,
                    "target_gripper": float(target[7]),
                    **metrics,
                }
                row["gripper_drift_abs"] = abs(row["gripper_drift"])
                row["gripper_sign_flip_vs_orig"] = int(row["gripper_sign"] != ref_sign)
                for i in range(8):
                    row[f"pred_a{i}"] = float(action[i])
                    row[f"orig_a{i}"] = float(ref[i])
                    row[f"target_a{i}"] = float(target[i])
                rows.append(row)

    csv_path = args.output_dir / "visual_perturbation_rows.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_count": len(indices),
        "indices": indices,
        "variants": variants,
        "train_stats_used": TRAIN_STATS,
        "eval_stats_used": EVAL_STATS,
        "summary_by_variant": summarize(rows, variants),
    }
    (args.output_dir / "visual_perturbation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
