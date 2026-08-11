#!/usr/bin/env python3
"""Audit visual-oneway-v2 data for a fair vision-versus-torque study.

This is deliberately stricter than a schema check.  It separates whether a
dataset is suitable to train the visual-localization baseline from whether it
contains *causal tactile supervision* for the subsequent insertion comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


PERCENTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
REQUIRED = {
    "state": (12,), "action": (6,), "rgb_table": None, "rgb_side": None,
    "joint_torque": (7,), "applied_wrench": (6,),
    "is_fine_visual_band": (1,), "audit_xy_error_m": (1,), "audit_depth_m": (1,),
}


def quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, np.float64).reshape(-1)
    if not len(x):
        return {}
    return {f"p{int(q * 100):02d}": float(v) for q, v in zip(PERCENTILES, np.quantile(x, PERCENTILES))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-demos", type=int, default=120)
    ap.add_argument("--min-fine-frames", type=int, default=8)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    all_xy: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    all_torque: list[np.ndarray] = []
    all_wrench: list[np.ndarray] = []
    all_fine: list[np.ndarray] = []
    image_means: list[float] = []
    image_stds: list[float] = []
    image_deltas: list[float] = []

    with h5py.File(args.input, "r") as f:
        attrs = {str(k): (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        if attrs.get("format") != "factory_peg_insert_visual_oneway_v2":
            errors.append(f"unexpected_format={attrs.get('format')!r}")
        if "demos" not in f:
            errors.append("missing demos group")
            demos: list[str] = []
        else:
            demos = sorted(f["demos"])
        if len(demos) != args.expected_demos:
            errors.append(f"demo_count={len(demos)}, expected={args.expected_demos}")

        for name in demos:
            g = f["demos"][name]
            missing = sorted(set(REQUIRED) - set(g))
            if missing:
                errors.append(f"{name}: missing={missing}")
                continue
            n = len(g["state"])
            shapes_ok = n > 0
            finite_ok = True
            for key, tail in REQUIRED.items():
                x = np.asarray(g[key])
                if len(x) != n or (tail is not None and x.shape[1:] != tail):
                    shapes_ok = False
                if key.startswith("rgb_"):
                    if x.dtype != np.uint8 or x.ndim != 4 or x.shape[-1] != 3:
                        shapes_ok = False
                    image_means.append(float(x.mean()))
                    image_stds.append(float(x.std()))
                    if len(x) > 1:
                        image_deltas.append(float(np.mean(np.abs(x[1:].astype(np.float32) - x[:-1].astype(np.float32)))))
                elif not np.isfinite(x).all():
                    finite_ok = False
            if not shapes_ok:
                errors.append(f"{name}: inconsistent shape/image schema")
            if not finite_ok:
                errors.append(f"{name}: non-finite numeric values")

            xy = np.asarray(g["audit_xy_error_m"], np.float32).reshape(-1)
            z = np.asarray(g["audit_depth_m"], np.float32).reshape(-1)
            fine = np.asarray(g["is_fine_visual_band"], bool).reshape(-1)
            action = np.asarray(g["action"], np.float32)
            torque = np.asarray(g["joint_torque"], np.float32)
            wrench = np.asarray(g["applied_wrench"], np.float32)
            fine_n = int(fine.sum())
            band_ok = fine_n >= args.min_fine_frames and bool(np.all((xy[fine] >= 0.001) & (xy[fine] <= 0.004)))
            if not band_ok:
                errors.append(f"{name}: invalid fine visual band ({fine_n} frames)")
            attr = g.attrs
            strict = bool(attr.get("strict_success", False))
            pre_action = attr.get("frame_alignment", "") == "pre_action"
            native = not bool(attr.get("state_intervention", True))
            if not (strict and pre_action and native):
                errors.append(f"{name}: strict={strict}, pre_action={pre_action}, native_reset={native}")
            rows.append({
                "demo": name, "frames": n, "fine_frames": fine_n, "strict_success": strict,
                "pre_action": pre_action, "native_reset": native,
                "initial_xy_error_m": float(xy[0]), "final_xy_error_m": float(xy[-1]),
                "initial_depth_m": float(z[0]), "final_depth_m": float(z[-1]),
                "fine_action_norm_median": float(np.median(np.linalg.norm(action[fine], axis=1))),
            })
            all_xy.append(xy); all_z.append(z); all_action.append(action); all_torque.append(torque); all_wrench.append(wrench); all_fine.append(fine)

    if not rows:
        errors.append("no auditable demonstrations")
        result = {"input": str(args.input), "errors": errors, "warnings": warnings}
    else:
        xy = np.concatenate(all_xy); z = np.concatenate(all_z); action = np.concatenate(all_action)
        torque = np.concatenate(all_torque); wrench = np.concatenate(all_wrench); fine = np.concatenate(all_fine)
        torque_norm = np.linalg.norm(torque, axis=1)
        wrench_norm = np.linalg.norm(wrench, axis=1)
        # The current expert is position-only.  Its source never branches on
        # torque/wrench, and this raw format has neither a contact/recovery
        # phase label nor paired visual states with different force outcomes.
        tactile_phase_labels = any("is_contact_recovery" in r or "is_recovery" in r for r in ())
        visual_ready = not errors and len(rows) >= args.expected_demos
        tactile_causal_ready = False
        if np.median(image_stds) < 5 or np.median(image_deltas) < 0.1:
            errors.append("images are degenerate or temporally static")
            visual_ready = False
        if np.std(torque_norm) < 1e-6:
            warnings.append("torque is numerically degenerate")
        warnings.extend([
            "The collector's oracle action is computed solely from held-vs-hole pose; it does not read torque or wrench.",
            "No contact/recovery-phase label or physically valid perturb-and-recovery branch is stored.",
            "Therefore this dataset is valid for visual-localization training, but is not causal tactile-supervision data and must not alone support a tactile-effect claim.",
        ])
        result = {
            "input": str(args.input), "raw_format": "factory_peg_insert_visual_oneway_v2",
            "integrity": {
                "demos": len(rows), "frames": int(len(xy)), "errors": errors,
                "all_strict_success": bool(all(r["strict_success"] for r in rows)),
                "all_pre_action": bool(all(r["pre_action"] for r in rows)),
                "all_native_reset": bool(all(r["native_reset"] for r in rows)),
            },
            "visual_localization": {
                "ready_for_training": bool(visual_ready),
                "fine_band_frames": int(fine.sum()), "fine_band_frame_fraction": float(fine.mean()),
                "fine_band_frames_per_demo": quantiles(np.array([r["fine_frames"] for r in rows])),
                "xy_error_m_all": quantiles(xy),
                "xy_error_m_initial": quantiles(np.array([r["initial_xy_error_m"] for r in rows])),
                "xy_error_m_final": quantiles(np.array([r["final_xy_error_m"] for r in rows])),
                "depth_m_all": quantiles(z),
                "action_norm_all": quantiles(np.linalg.norm(action, axis=1)),
                "action_norm_fine_band": quantiles(np.linalg.norm(action[fine], axis=1)),
                "image_mean": quantiles(np.array(image_means)), "image_std": quantiles(np.array(image_stds)),
                "image_temporal_abs_delta": quantiles(np.array(image_deltas)),
            },
            "torque_signal": {
                "joint_torque_norm": quantiles(torque_norm), "applied_wrench_norm": quantiles(wrench_norm),
                "fine_band_joint_torque_norm": quantiles(torque_norm[fine]),
                "fine_band_applied_wrench_norm": quantiles(wrench_norm[fine]),
                "non_degenerate": bool(np.std(torque_norm) >= 1e-6),
            },
            "tactile_causal_comparison": {
                "ready_for_training": tactile_causal_ready,
                "phase_labels_present": tactile_phase_labels,
                "reason": "Position-only oracle labels provide no torque-conditioned corrective action; collect a native-reset, physically valid near-rim contact/recovery supplement before training the torque comparison.",
            },
            "warnings": warnings, "per_demo": rows,
        }

    out_json = args.output_dir / "audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    integ = result.get("integrity", {})
    vis = result.get("visual_localization", {})
    tactile = result.get("tactile_causal_comparison", {})
    lines = [
        "# Visual Oneway v2 完整质量审计", "",
        f"- 原始数据：`{args.input}`", f"- 演示/帧数：{integ.get('demos', 0)} / {integ.get('frames', 0)}",
        f"- 视觉定位训练：{'**通过**' if vis.get('ready_for_training') else '**不通过**'}",
        f"- 触觉因果对照训练：{'**通过**' if tactile.get('ready_for_training') else '**不通过**'}", "",
        "## 结论", "",
        "该数据集可作为严格、无状态干预的视觉定位训练集；但不能单独作为证明 LSTM 力矩有效性的训练集。",
        "原因是专家动作只依赖位姿误差，未出现“视觉近似相同、受力不同、正确动作不同”的接触恢复样本。",
        "", "## 指标", "",
        f"- 严格成功：{integ.get('all_strict_success')}; 预动作帧：{integ.get('all_pre_action')}; 原生重置：{integ.get('all_native_reset')}",
        f"- 近孔 1–4 mm 帧数：{vis.get('fine_band_frames')} ({vis.get('fine_band_frame_fraction', 0):.1%})",
        f"- 每轨近孔帧分位数：{vis.get('fine_band_frames_per_demo')}",
        f"- 力矩非退化：{result.get('torque_signal', {}).get('non_degenerate')}",
        "", "## 训练准入", "",
        "1. 现在可训练纯视觉定位基线，并保留固定的 20 条轨迹作验证集。",
        "2. 不应把本数据直接转换为“视觉 vs LSTM 力矩”的最终结论性四组训练。",
        "3. 需补充原生重置下的近孔接触/恢复轨迹：控制视觉初始误差、注入物理横向扰动或倾角，记录接触阶段，且专家在受力后采取实际不同的退让/横向重对齐动作。随后视觉、原始力矩、零力矩、因果打乱力矩使用同一训练轨迹与种子。",
    ]
    if result.get("warnings"):
        lines += ["", "## 限制"] + [f"- {x}" for x in result["warnings"]]
    if integ.get("errors"):
        lines += ["", "## 错误"] + [f"- {x}" for x in integ["errors"]]
    (args.output_dir / "QUALITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"visual_ready": vis.get("ready_for_training"), "tactile_ready": tactile.get("ready_for_training"), "output": str(out_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
