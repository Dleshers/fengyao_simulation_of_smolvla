#!/usr/bin/env python3
"""Paired visual-difference audit between eval snapshots and nearest train frames.

The audit pairs saved eval policy-input snapshots with nearest-neighbor training
frames found by dataset_nearest_eval_state_audit.py.  It computes simple,
interpretable image statistics and writes side-by-side montages.

This is offline-only and does not train, evaluate, or modify assets.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFilter


def pil_to_arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def load_eval_image(snapshot_dir: Path, step: int, camera: str) -> Image.Image:
    path = snapshot_dir / f"step_{step:04d}" / f"observation_images_{camera}.png"
    return Image.open(path).convert("RGB")


def has_eval_images(snapshot_dir: Path, step: int) -> bool:
    step_dir = snapshot_dir / f"step_{step:04d}"
    return (step_dir / "observation_images_camera1.png").exists() and (
        step_dir / "observation_images_camera2.png"
    ).exists()


def load_train_image(parquet_path: Path, local_row: int, key: str) -> Image.Image:
    table = pq.read_table(parquet_path, columns=[key])
    row = table.slice(local_row, 1).to_pylist()[0][key]
    return Image.open(io.BytesIO(row["bytes"])).convert("RGB")


def rgb_to_hsv_like(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    val = mx
    return sat, val


def image_metrics(img: Image.Image) -> dict[str, float | list[float] | None]:
    arr = pil_to_arr(img)
    gray = arr.mean(axis=-1)
    sat, val = rgb_to_hsv_like(arr)
    edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0

    bright = gray > 0.85
    whiteish = (val > 0.80) & (sat < 0.18)
    dark = gray < 0.10
    saturated = (sat > 0.35) & (val > 0.18)

    metrics: dict[str, float | list[float] | None] = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "sat_mean": float(sat.mean()),
        "sat_p95": float(np.percentile(sat, 95)),
        "val_mean": float(val.mean()),
        "bright_frac": float(bright.mean()),
        "whiteish_frac": float(whiteish.mean()),
        "dark_frac": float(dark.mean()),
        "edge_mean": float(edges.mean()),
        "edge_p95": float(np.percentile(edges, 95)),
        "saturated_frac": float(saturated.mean()),
    }

    if saturated.any():
        ys, xs = np.nonzero(saturated)
        h, w = gray.shape
        metrics.update(
            {
                "sat_centroid_x": float(xs.mean() / max(w - 1, 1)),
                "sat_centroid_y": float(ys.mean() / max(h - 1, 1)),
                "sat_bbox": [
                    float(xs.min() / max(w - 1, 1)),
                    float(ys.min() / max(h - 1, 1)),
                    float(xs.max() / max(w - 1, 1)),
                    float(ys.max() / max(h - 1, 1)),
                ],
            }
        )
    else:
        metrics.update({"sat_centroid_x": None, "sat_centroid_y": None, "sat_bbox": None})

    # Coarse spatial brightness layout: left/right/top/bottom and center patch.
    h, w = gray.shape
    metrics.update(
        {
            "gray_left_mean": float(gray[:, : w // 2].mean()),
            "gray_right_mean": float(gray[:, w // 2 :].mean()),
            "gray_top_mean": float(gray[: h // 2, :].mean()),
            "gray_bottom_mean": float(gray[h // 2 :, :].mean()),
            "gray_center_mean": float(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()),
        }
    )
    return metrics


def image_diff_metrics(eval_img: Image.Image, train_img: Image.Image) -> dict[str, float]:
    e = pil_to_arr(eval_img.resize((224, 224)))
    t = pil_to_arr(train_img.resize((224, 224)))
    diff = e - t
    return {
        "l1_mean": float(np.abs(diff).mean()),
        "l2_mean": float(np.sqrt((diff * diff).mean())),
        "linf": float(np.abs(diff).max()),
        "corr_gray": float(np.corrcoef(e.mean(axis=-1).reshape(-1), t.mean(axis=-1).reshape(-1))[0, 1]),
    }


def label_tile(img: Image.Image, label: str) -> Image.Image:
    out = img.convert("RGB").resize((224, 224))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, 224, 28), fill=(255, 255, 255))
    draw.text((4, 6), label, fill=(0, 0, 0))
    return out


def save_montage(rows: list[dict], output: Path) -> None:
    tiles: list[Image.Image] = []
    for r in rows:
        step = r["step"]
        train_ep = r["episode_index"]
        train_frame = r["frame_index"]
        tiles.extend(
            [
                label_tile(r["_eval_camera1"], f"eval s{step} cam1"),
                label_tile(r["_train_camera1"], f"train ep{train_ep} f{train_frame} cam1"),
                label_tile(r["_eval_camera2"], f"eval s{step} cam2"),
                label_tile(r["_train_camera2"], f"train ep{train_ep} f{train_frame} cam2"),
            ]
        )
    cols = 4
    canvas = Image.new("RGB", (cols * 224, math.ceil(len(tiles) / cols) * 224), (235, 235, 235))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % cols) * 224, (i // cols) * 224))
    canvas.save(output, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nearest-csv", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nearest_rows = []
    with args.nearest_csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["rank"]) == args.rank:
                nearest_rows.append(row)

    output_rows = []
    detailed_rows = []
    for row in nearest_rows:
        step = int(row["eval_step"])
        if not has_eval_images(args.snapshot_dir, step):
            continue
        parquet_path = Path(row["parquet"])
        local_row = int(row["local_row"])

        eval_c1 = load_eval_image(args.snapshot_dir, step, "camera1")
        eval_c2 = load_eval_image(args.snapshot_dir, step, "camera2")
        train_c1 = load_train_image(parquet_path, local_row, "observation.images.camera1")
        train_c2 = load_train_image(parquet_path, local_row, "observation.images.camera2")

        base = {
            "step": step,
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "distance_state_z": float(row["distance_state_z"]),
            "eval_gripper_cmd": float(row["eval_gripper_cmd"]),
            "train_action_gripper": float(row["train_action_gripper"]),
            "eval_eef_cube_dist_after": float(row["eval_eef_cube_dist_after"]),
        }
        compact = dict(base)
        for cam, e_img, t_img in [("camera1", eval_c1, train_c1), ("camera2", eval_c2, train_c2)]:
            e_metrics = image_metrics(e_img)
            t_metrics = image_metrics(t_img)
            d_metrics = image_diff_metrics(e_img, t_img)
            for key in [
                "mean",
                "std",
                "sat_mean",
                "bright_frac",
                "whiteish_frac",
                "edge_mean",
                "saturated_frac",
                "sat_centroid_x",
                "sat_centroid_y",
                "gray_left_mean",
                "gray_right_mean",
                "gray_top_mean",
                "gray_bottom_mean",
                "gray_center_mean",
            ]:
                compact[f"{cam}_eval_{key}"] = e_metrics[key]
                compact[f"{cam}_train_{key}"] = t_metrics[key]
                if isinstance(e_metrics[key], (int, float)) and isinstance(t_metrics[key], (int, float)):
                    compact[f"{cam}_delta_{key}"] = float(e_metrics[key]) - float(t_metrics[key])
            for key, val in d_metrics.items():
                compact[f"{cam}_diff_{key}"] = val
        output_rows.append(compact)
        detailed = dict(compact)
        detailed["_eval_camera1"] = eval_c1
        detailed["_eval_camera2"] = eval_c2
        detailed["_train_camera1"] = train_c1
        detailed["_train_camera2"] = train_c2
        detailed_rows.append(detailed)

    csv_path = args.output_dir / "paired_visual_diff_rows.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    # Aggregate numeric columns.
    aggregate = {}
    numeric_keys = [
        k for k, v in output_rows[0].items() if isinstance(v, (int, float)) and k not in {"step", "episode_index", "frame_index"}
    ]
    for key in numeric_keys:
        vals = np.asarray([r[key] for r in output_rows if r[key] is not None], dtype=np.float64)
        if vals.size:
            aggregate[key] = {
                "mean": float(vals.mean()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

    summary = {"rank": args.rank, "n_pairs": len(output_rows), "aggregate": aggregate, "rows": output_rows}
    (args.output_dir / "paired_visual_diff_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_montage(detailed_rows, args.output_dir / "paired_eval_train_montage.jpg")

    # Print a compact human-readable focus.
    focus = []
    for r in output_rows:
        focus.append(
            {
                "step": r["step"],
                "eef_cube": r["eval_eef_cube_dist_after"],
                "cam2_mean_delta": r["camera2_delta_mean"],
                "cam2_whiteish_delta": r["camera2_delta_whiteish_frac"],
                "cam2_sat_centroid_eval": [r["camera2_eval_sat_centroid_x"], r["camera2_eval_sat_centroid_y"]],
                "cam2_sat_centroid_train": [r["camera2_train_sat_centroid_x"], r["camera2_train_sat_centroid_y"]],
                "cam2_l1": r["camera2_diff_l1_mean"],
            }
        )
    print(json.dumps(focus, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
