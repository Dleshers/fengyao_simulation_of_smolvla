#!/usr/bin/env python3
"""Find nearest training frames to selected eval trajectory states.

This is an offline diagnostic.  It compares raw joint-state observations from an
eval trajectory JSONL against the LeRobot parquet dataset, then reports nearest
training actions and image statistics.  It does not train or modify assets.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def load_dataset(dataset_root: Path) -> dict:
    states = []
    actions = []
    rows = []
    image_refs = []
    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "observation.images.camera1",
                "observation.images.camera2",
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
            ],
        )
        for local_i, row in enumerate(table.to_pylist()):
            states.append(row["observation.state"])
            actions.append(row["action"])
            rows.append(
                {
                    "parquet": str(parquet_path),
                    "local_row": local_i,
                    "index": row["index"],
                    "episode_index": row["episode_index"],
                    "frame_index": row["frame_index"],
                    "timestamp": row["timestamp"],
                }
            )
            image_refs.append(
                {
                    "camera1": row["observation.images.camera1"],
                    "camera2": row["observation.images.camera2"],
                }
            )
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rows": rows,
        "image_refs": image_refs,
    }


def load_eval_targets(traj_path: Path, steps: list[int]) -> dict[int, dict]:
    wanted = set(steps)
    out = {}
    for line in traj_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") != "step" or int(rec["step"]) not in wanted:
            continue
        state = list(rec["joint_pos_after"]) + list(rec["gripper_qpos_after"])
        out[int(rec["step"])] = {"state": np.asarray(state, dtype=np.float32), "record": rec}
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"missing target steps in trajectory: {missing}")
    return out


def image_stats(image_struct: dict) -> dict:
    img = Image.open(io.BytesIO(image_struct["bytes"])).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def image_from_struct(image_struct: dict, label: str) -> Image.Image:
    img = Image.open(io.BytesIO(image_struct["bytes"])).convert("RGB").resize((224, 224))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 224, 24), fill=(255, 255, 255))
    draw.text((4, 4), label, fill=(0, 0, 0))
    return img


def save_montage(matches: list[dict], output_path: Path, max_rows: int = 24) -> None:
    tiles = []
    for m in matches[:max_rows]:
        label = f"s{m['eval_step']} k{m['rank']} ep{m['episode_index']} f{m['frame_index']}"
        tiles.append(image_from_struct(m["_image_refs"]["camera1"], label + " cam1"))
        tiles.append(image_from_struct(m["_image_refs"]["camera2"], label + " cam2"))
    if not tiles:
        return
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * 224, rows * 224), (240, 240, 240))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % cols) * 224, (i // cols) * 224))
    canvas.save(output_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", default="55,57,59,60,61,62,63,64,65,66,70")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    data = load_dataset(args.dataset_root)
    targets = load_eval_targets(args.trajectory, steps)

    stats = json.loads((args.dataset_root / "meta" / "stats.json").read_text())
    state_std = np.asarray(stats["observation.state"]["std"], dtype=np.float32)
    state_std = np.maximum(state_std, 1e-6)

    all_matches = []
    compact = {}
    for step in steps:
        target = targets[step]
        diff = (data["states"] - target["state"][None, :]) / state_std[None, :]
        d_all = np.linalg.norm(diff, axis=1)
        d_arm = np.linalg.norm(diff[:, :7], axis=1)
        nearest = np.argsort(d_all)[: args.top_k]
        compact[str(step)] = []
        for rank, idx in enumerate(nearest, start=1):
            row_meta = data["rows"][int(idx)]
            img_refs = data["image_refs"][int(idx)]
            cam1_stats = image_stats(img_refs["camera1"])
            cam2_stats = image_stats(img_refs["camera2"])
            action = data["actions"][int(idx)]
            state = data["states"][int(idx)]
            rec = target["record"]
            match = {
                "eval_step": step,
                "rank": rank,
                "distance_state_z": float(d_all[int(idx)]),
                "distance_arm_z": float(d_arm[int(idx)]),
                "parquet": row_meta["parquet"],
                "local_row": row_meta["local_row"],
                "dataset_index": row_meta["index"],
                "episode_index": row_meta["episode_index"],
                "frame_index": row_meta["frame_index"],
                "timestamp": row_meta["timestamp"],
                "eval_gripper_cmd": float(rec["action_gripper_cmd"]),
                "train_action_gripper": float(action[7]),
                "eval_eef_cube_dist_after": float(rec["eef_cube_dist_after"]),
                "eval_joint_target_error_l2": float(rec["joint_target_error_l2"]),
                "train_camera1_mean": cam1_stats["mean"],
                "train_camera1_std": cam1_stats["std"],
                "train_camera2_mean": cam2_stats["mean"],
                "train_camera2_std": cam2_stats["std"],
                "eval_state": target["state"].tolist(),
                "train_state": state.tolist(),
                "train_action": action.tolist(),
                "_image_refs": img_refs,
            }
            all_matches.append(match)
            compact[str(step)].append({k: v for k, v in match.items() if not k.startswith("_")})

    csv_fields = [
        "eval_step",
        "rank",
        "distance_state_z",
        "distance_arm_z",
        "parquet",
        "local_row",
        "dataset_index",
        "episode_index",
        "frame_index",
        "timestamp",
        "eval_gripper_cmd",
        "train_action_gripper",
        "eval_eef_cube_dist_after",
        "eval_joint_target_error_l2",
        "train_camera1_mean",
        "train_camera1_std",
        "train_camera2_mean",
        "train_camera2_std",
    ]
    with (args.output_dir / "nearest_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for m in all_matches:
            writer.writerow({k: m[k] for k in csv_fields})

    (args.output_dir / "nearest_summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    save_montage(all_matches, args.output_dir / "nearest_camera_montage.jpg")

    print(json.dumps({k: compact[k][:2] for k in list(compact)[:3]}, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
