#!/usr/bin/env python3
"""Replay saved eval policy-input snapshots with camera ablations.

This is offline-only: no Isaac, no training. It reconstructs already-preprocessed
policy inputs from snapshot.json + PNG files, runs the local baseline, then
compares the postprocessed action against the action recorded during eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


TRAIN_STATS = {
    "camera1": {"mean": 0.7332273721694946, "std": 0.2509204149246216},
    "camera2": {"mean": 0.441604882478714, "std": 0.20611362159252167},
}


def load_png(path: Path, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    arr = arr.view(img.height, img.width, 3).float() / 255.0
    return arr.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def affine_match(img: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    cur_mean = img.mean()
    cur_std = img.std(unbiased=False).clamp_min(1e-6)
    return ((img - cur_mean) / cur_std * std + mean).clamp(0.0, 1.0)


def build_batch(step_dir: Path, device: str) -> tuple[dict, torch.Tensor]:
    snap = json.loads((step_dir / "snapshot.json").read_text())
    po = snap["policy_observation"]

    def tensor_from_head(key: str, dtype: torch.dtype) -> torch.Tensor:
        shape = po[key]["shape"]
        data = po[key]["head"]
        return torch.tensor(data, dtype=dtype, device=device).view(*shape)

    batch = {
        "observation.state": tensor_from_head("observation.state", torch.float32),
        "observation.images.camera1": load_png(step_dir / "observation_images_camera1.png", device),
        "observation.images.camera2": load_png(step_dir / "observation_images_camera2.png", device),
        "observation.tactile.force_grid": torch.zeros((1, 2, 10, 12, 3), dtype=torch.float32, device=device),
        "observation.language.tokens": tensor_from_head("observation.language.tokens", torch.long),
        "observation.language.attention_mask": tensor_from_head(
            "observation.language.attention_mask", torch.bool
        ),
        "task": po.get("task", ["Pick and place the cube into the basket\n"]),
    }
    recorded = torch.tensor(snap["env_action"]["head"], dtype=torch.float32)
    return batch, recorded


def apply_variant(batch: dict, variant: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.clone() if isinstance(v, torch.Tensor) else v
    c1 = "observation.images.camera1"
    c2 = "observation.images.camera2"
    if variant == "orig":
        return out
    if variant == "cam2_train_affine":
        out[c2] = affine_match(out[c2], TRAIN_STATS["camera2"]["mean"], TRAIN_STATS["camera2"]["std"])
    elif variant == "cam2_zero":
        out[c2] = torch.zeros_like(out[c2])
    elif variant == "cam2_flat_train_mean":
        out[c2] = torch.full_like(out[c2], TRAIN_STATS["camera2"]["mean"])
    elif variant == "cam1_train_affine":
        out[c1] = affine_match(out[c1], TRAIN_STATS["camera1"]["mean"], TRAIN_STATS["camera1"]["std"])
    elif variant == "both_train_affine":
        out[c1] = affine_match(out[c1], TRAIN_STATS["camera1"]["mean"], TRAIN_STATS["camera1"]["std"])
        out[c2] = affine_match(out[c2], TRAIN_STATS["camera2"]["mean"], TRAIN_STATS["camera2"]["std"])
    elif variant == "swap_cameras":
        out[c1], out[c2] = out[c2].clone(), out[c1].clone()
    else:
        raise ValueError(variant)
    return out


def action_row(step: int, variant: str, action: torch.Tensor, ref: torch.Tensor, recorded: torch.Tensor) -> dict:
    action = action.float().cpu().flatten()
    ref = ref.float().cpu().flatten()
    recorded = recorded.float().cpu().flatten()
    drift = action - ref
    rec_delta = action - recorded
    return {
        "step": step,
        "variant": variant,
        "arm_drift_l2_vs_orig": float(torch.linalg.vector_norm(drift[:7])),
        "arm_drift_linf_vs_orig": float(drift[:7].abs().max()),
        "gripper_drift_vs_orig": float(drift[7]),
        "gripper_pred": float(action[7]),
        "gripper_sign": -1 if float(action[7]) < 0 else 1,
        "arm_l2_vs_recorded": float(torch.linalg.vector_norm(rec_delta[:7])),
        "gripper_delta_vs_recorded": float(rec_delta[7]),
        **{f"pred_a{i}": float(action[i]) for i in range(8)},
        **{f"recorded_a{i}": float(recorded[i]) for i in range(8)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds_meta = LeRobotDatasetMetadata("franka_pickplace_joint_visual_torque_w30_v1", root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = args.device
    cfg.load_vlm_weights = False
    cfg.n_action_steps = 1
    cfg.use_torque_lstm = False

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    _, postprocessor = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(args.policy_path))

    variants = [
        "orig",
        "cam2_train_affine",
        "cam2_zero",
        "cam2_flat_train_mean",
        "cam1_train_affine",
        "both_train_affine",
        "swap_cameras",
    ]

    rows = []
    with torch.no_grad():
        for step_dir in sorted(args.snapshot_dir.glob("step_*")):
            step = int(step_dir.name.split("_")[1])
            batch, recorded = build_batch(step_dir, args.device)
            predictions = {}
            for variant in variants:
                policy.reset()
                out = policy.select_action(apply_variant(batch, variant))
                action = postprocessor(out).squeeze(0).cpu()
                predictions[variant] = action
            ref = predictions["orig"]
            for variant in variants:
                rows.append(action_row(step, variant, predictions[variant], ref, recorded))

    import csv

    csv_path = args.output_dir / "snapshot_ablation_rows.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"variants": variants, "rows": rows}
    (args.output_dir / "snapshot_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
