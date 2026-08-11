#!/usr/bin/env python3
"""Admission audit for native physical contact-recovery demonstrations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED = {
    "state": (12,), "action": (6,), "rgb_table": None, "rgb_side": None,
    "joint_torque": (7,), "applied_wrench": (6,), "phase": (1,),
    "is_policy_label": (1,), "audit_xy_error_m": (1,), "audit_depth_m": (1,),
}


def q(x):
    x = np.asarray(x, np.float64).reshape(-1)
    return {f"p{int(p * 100):02d}": float(v) for p, v in zip((0, .05, .5, .95, 1), np.quantile(x, (0, .05, .5, .95, 1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-demos", type=int, default=64)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    errors, rows = [], []
    with h5py.File(args.input, "r") as f:
        if f.attrs.get("format", "") not in ("factory_peg_insert_contact_recovery_native_v1", "factory_peg_insert_contact_recovery_native_v1_balanced64"): errors.append("unexpected raw format")
        demos = sorted(f.get("demos", {}))
        if len(demos) != args.expected_demos: errors.append(f"demo_count={len(demos)}, expected={args.expected_demos}")
        for name in demos:
            g = f["demos"][name]; missing = sorted(set(REQUIRED) - set(g))
            if missing: errors.append(f"{name}: missing={missing}"); continue
            n = len(g["state"]); shape_ok = n > 0; finite_ok = True
            for key, tail in REQUIRED.items():
                x = np.asarray(g[key])
                shape_ok &= len(x) == n and (tail is None or x.shape[1:] == tail)
                if key not in ("rgb_table", "rgb_side", "phase", "is_policy_label"): finite_ok &= bool(np.isfinite(x).all())
            a = g.attrs
            xy, z, td = float(a.get("contact_xy_error_m", np.nan)), float(a.get("contact_depth_m", np.nan)), float(a.get("contact_torque_delta", np.nan))
            labels, hist, dot = int(a.get("recovery_label_frames", 0)), int(a.get("contact_history_frames", 0)), float(a.get("recenter_direction_dot", np.nan))
            final_xy, final_z = float(a.get("final_xy_error_m", np.nan)), float(a.get("final_depth_m", np.nan))
            valid = shape_ok and finite_ok and bool(a.get("strict_success", False)) and a.get("frame_alignment", "") == "pre_action" and not bool(a.get("state_intervention", True)) and .0025 <= xy <= .009 and .015 <= z <= .032 and td >= .03 and hist >= 30 and labels >= 20 and dot >= .70 and final_xy < .0025 and final_z < .001
            if not valid: errors.append(f"{name}: failed physical-contact admission gate")
            rows.append({"demo": name, "valid": bool(valid), "direction_sector": int(a.get("direction_sector", -1)), "frames": n, "contact_xy_m": xy, "contact_z_m": z, "torque_delta": td, "history_frames": hist, "label_frames": labels, "direction_dot": dot, "final_xy_m": final_xy, "final_z_m": final_z})
    sectors = {str(i): sum(r["direction_sector"] == i for r in rows) for i in range(8)}
    expected_per_sector = args.expected_demos // 8
    if args.expected_demos % 8 or any(v != expected_per_sector for v in sectors.values()): errors.append(f"unbalanced direction coverage: {sectors}; expected {expected_per_sector} each")
    result = {"input": str(args.input), "demos": len(rows), "frames": int(sum(r["frames"] for r in rows)), "all_valid": not errors and bool(rows), "errors": errors, "direction_counts": sectors, "contact_xy_m": q([r["contact_xy_m"] for r in rows]) if rows else {}, "contact_z_m": q([r["contact_z_m"] for r in rows]) if rows else {}, "torque_delta": q([r["torque_delta"] for r in rows]) if rows else {}, "recovery_label_frames": q([r["label_frames"] for r in rows]) if rows else {}, "rows": rows}
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Native Contact-Recovery Gate-B Audit", "", f"- Demos: {result['demos']}; frames: {result['frames']}", f"- Admission: {'PASS' if result['all_valid'] else 'FAIL'}", f"- Direction counts: {sectors}", f"- Contact XY (m): {result['contact_xy_m']}", f"- Contact height (m): {result['contact_z_m']}", f"- Torque excursion: {result['torque_delta']}", f"- Recovery-label frames: {result['recovery_label_frames']}", "", "## Decision", "", "This contact set is admitted only when every trajectory has native-reset, pre-action alignment, an uninserted rim-blocked state, a 30-step real torque history, torque excursion, and strict physical recovery.", "", "## Errors"] + [f"- {e}" for e in errors]
    (args.output_dir / "QUALITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"all_valid": result["all_valid"], "demos": result["demos"], "output": str(args.output_dir)}))


if __name__ == "__main__": main()
