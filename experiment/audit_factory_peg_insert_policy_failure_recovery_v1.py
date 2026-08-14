#!/usr/bin/env python3
"""Admission audit for actual-policy sub-millimetre recovery demonstrations."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

REQUIRED = {
    "state": (12,),
    "action": (6,),
    "rgb_table": None,
    "rgb_side": None,
    "joint_torque": (7,),
    "applied_wrench": (6,),
    "phase": (1,),
    "is_policy_label": (1,),
    "audit_xy_error_m": (1,),
    "audit_depth_m": (1,),
}


def quantiles(values):
    x = np.asarray(values, np.float64)
    if not len(x):
        return {}
    points = (0.0, 0.05, 0.5, 0.95, 1.0)
    return {f"p{int(p * 100):02d}": float(v) for p, v in zip(points, np.quantile(x, points))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-demos", type=int, default=16)
    parser.add_argument("--require-balanced-grid", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    errors, rows = [], []
    seen_seeds, seen_pairs = set(), set()
    with h5py.File(args.input, "r") as h5:
        if h5.attrs.get("format", "") != "factory_peg_insert_policy_failure_recovery_v1":
            errors.append(f"unexpected format={h5.attrs.get('format', '')!r}")
        demos = sorted(h5.get("demos", {}))
        if len(demos) != args.expected_demos:
            errors.append(f"demo_count={len(demos)}, expected={args.expected_demos}")
        for name in demos:
            group = h5["demos"][name]
            missing = sorted(set(REQUIRED) - set(group))
            if missing:
                errors.append(f"{name}: missing={missing}")
                continue
            n = len(group["state"])
            shape_ok, finite_ok = n > 0, True
            for key, tail in REQUIRED.items():
                value = np.asarray(group[key])
                shape_ok &= len(value) == n and (tail is None or value.shape[1:] == tail)
                if key not in ("rgb_table", "rgb_side", "phase", "is_policy_label"):
                    finite_ok &= bool(np.isfinite(value).all())

            attr = group.attrs
            labels = np.asarray(group["is_policy_label"], bool).reshape(-1)
            phase = np.asarray(group["phase"], np.uint8).reshape(-1)
            xy_trace = np.asarray(group["audit_xy_error_m"], np.float64).reshape(-1)
            z_trace = np.asarray(group["audit_depth_m"], np.float64).reshape(-1)
            seed = int(attr.get("episode_seed", -1))
            pair_id = str(attr.get("pair_id", ""))
            sector = int(attr.get("direction_sector", -1))
            load = int(attr.get("load_band", -1))
            contact_xy = float(attr.get("contact_xy_error_m", np.nan))
            contact_z = float(attr.get("contact_depth_m", np.nan))
            torque_delta = float(attr.get("contact_torque_delta", np.nan))
            failure_xy = float(attr.get("failure_xy_error_m", np.nan))
            failure_z = float(attr.get("failure_depth_m", np.nan))
            policy_min_xy = float(attr.get("policy_min_xy_error_m", np.nan))
            final_xy = float(attr.get("final_xy_error_m", np.nan))
            final_z = float(attr.get("final_depth_m", np.nan))
            recovery_start = int(attr.get("recovery_start_frame", -1))
            label_frames = int(attr.get("recovery_label_frames", -1))
            hold_steps = int(attr.get("strict_hold_steps", -1))

            row_errors = []
            if not shape_ok:
                row_errors.append("shape")
            if not finite_ok:
                row_errors.append("nonfinite")
            if not bool(attr.get("strict_success", False)):
                row_errors.append("not_strict_success")
            if attr.get("frame_alignment", "") != "pre_action":
                row_errors.append("frame_alignment")
            if bool(attr.get("state_intervention", True)):
                row_errors.append("state_intervention")
            if str(attr.get("source_kind", "")) != "policy_failure":
                row_errors.append("source_kind")
            if str(attr.get("policy_arm", "")) != "hybrid_visual_torque":
                row_errors.append("policy_arm")
            if not (0.0025 <= contact_xy <= 0.009 and 0.015 <= contact_z <= 0.032):
                row_errors.append("native_contact")
            if torque_delta < 0.02 or int(attr.get("contact_history_frames", 0)) != 30:
                row_errors.append("contact_torque_history")
            if not (0.0 <= failure_xy < 0.001 and 0.015 <= failure_z <= 0.032):
                row_errors.append("failure_not_blocked_submm")
            if not (0.0 <= policy_min_xy < 0.001):
                row_errors.append("policy_never_visited_submm")
            if int(attr.get("policy_steps", 0)) < 2 or not str(attr.get("policy_failure_reason", "")):
                row_errors.append("missing_policy_failure")
            if label_frames != int(labels.sum()) or label_frames < 45:
                row_errors.append("recovery_labels")
            if recovery_start != int(np.flatnonzero(labels)[0]) if labels.any() else True:
                row_errors.append("recovery_start")
            if labels.any() and (np.any(phase[labels] < 5) or np.any(phase[~labels] >= 5)):
                row_errors.append("label_phase_contract")
            if hold_steps < 10 or int(np.sum(phase == 8)) < 10:
                row_errors.append("strict_hold_count")
            hold = phase == 8
            if hold.any() and not bool(np.all((xy_trace[hold] < 0.0025) & (z_trace[hold] >= -0.002) & (z_trace[hold] <= 0.001))):
                row_errors.append("strict_hold_trace")
            if not (final_xy < 0.0025 and -0.002 <= final_z <= 0.001):
                row_errors.append("final_strict")
            if seed in seen_seeds or pair_id in seen_pairs or not pair_id:
                row_errors.append("duplicate_identity")
            seen_seeds.add(seed)
            seen_pairs.add(pair_id)
            if not (0 <= sector < 8 and load in (0, 1)):
                row_errors.append("grid_cell")

            if row_errors:
                errors.append(f"{name}: {','.join(row_errors)}")
            rows.append({
                "demo": name,
                "valid": not row_errors,
                "seed": seed,
                "pair_id": pair_id,
                "sector": sector,
                "load_band": load,
                "frames": n,
                "policy_steps": int(attr.get("policy_steps", 0)),
                "failure_reason": str(attr.get("policy_failure_reason", "")),
                "contact_xy_m": contact_xy,
                "contact_z_m": contact_z,
                "torque_delta": torque_delta,
                "policy_min_xy_m": policy_min_xy,
                "failure_xy_m": failure_xy,
                "failure_z_m": failure_z,
                "label_frames": label_frames,
                "final_xy_m": final_xy,
                "final_z_m": final_z,
            })

    grid = Counter((row["sector"], row["load_band"]) for row in rows)
    if args.require_balanced_grid:
        if args.expected_demos % 16:
            errors.append("balanced grid requires expected demos divisible by 16")
        else:
            expected = args.expected_demos // 16
            bad = {f"s{s}_l{l}": grid[(s, l)] for s in range(8) for l in range(2) if grid[(s, l)] != expected}
            if bad:
                errors.append(f"unbalanced sector/load grid={bad}; expected={expected}")

    result = {
        "input": str(args.input),
        "expected_demos": args.expected_demos,
        "demos": len(rows),
        "frames": int(sum(row["frames"] for row in rows)),
        "all_valid": bool(rows) and not errors,
        "errors": errors,
        "grid_counts": {f"s{s}_l{l}": grid[(s, l)] for s in range(8) for l in range(2)},
        "contact_xy_m": quantiles([row["contact_xy_m"] for row in rows]),
        "failure_xy_m": quantiles([row["failure_xy_m"] for row in rows]),
        "failure_z_m": quantiles([row["failure_z_m"] for row in rows]),
        "policy_min_xy_m": quantiles([row["policy_min_xy_m"] for row in rows]),
        "torque_delta": quantiles([row["torque_delta"] for row in rows]),
        "recovery_label_frames": quantiles([row["label_frames"] for row in rows]),
        "rows": rows,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Actual-policy Sub-mm Recovery Audit",
        "",
        f"- Admission: {'PASS' if result['all_valid'] else 'FAIL'}",
        f"- Demos / frames: {result['demos']} / {result['frames']}",
        f"- Sector-load grid: {result['grid_counts']}",
        f"- Failure XY (m): {result['failure_xy_m']}",
        f"- Failure depth (m): {result['failure_z_m']}",
        f"- Torque excursion: {result['torque_delta']}",
        f"- Recovery labels: {result['recovery_label_frames']}",
        "",
        "Every admitted episode is native-reset, physically contacted, visited by the frozen hybrid policy, blocked below 1 mm, and strictly recovered without pose writes.",
        "",
        "## Errors",
    ]
    lines.extend(f"- {error}" for error in errors)
    (args.output_dir / "QUALITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"all_valid": result["all_valid"], "demos": len(rows), "output": str(args.output_dir)}))
    raise SystemExit(0 if result["all_valid"] else 2)


if __name__ == "__main__":
    main()
