#!/usr/bin/env python3
"""Materialize an exactly 8-by-8 balanced contact-recovery HDF5 and manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


ap = argparse.ArgumentParser()
ap.add_argument("--main", type=Path, required=True, help="64-demo source with 9 demos except sector 4")
ap.add_argument("--sector4-smoke", type=Path, required=True)
ap.add_argument("--sector4-formal", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--manifest", type=Path, required=True)
a = ap.parse_args()

sources = [("main", a.main), ("sector4_smoke", a.sector4_smoke), ("sector4_formal", a.sector4_formal)]
entries = []
for source_name, path in sources:
    with h5py.File(path, "r") as f:
        for name in sorted(f["demos"]):
            g = f["demos"][name]
            if not bool(g.attrs.get("strict_success", False)):
                continue
            entries.append({"source": source_name, "path": str(path), "demo": name, "sector": int(g.attrs["direction_sector"]), "contact_xy_m": float(g.attrs["contact_xy_error_m"]), "torque_delta": float(g.attrs["contact_torque_delta"])})

selected = []
for sector in range(8):
    candidates = [e for e in entries if e["sector"] == sector]
    # For sectors 0..3 and 5..7, retain the first eight main trajectories.
    # Sector 4 deliberately uses its one main trajectory plus 2+5 calibrated
    # trajectories, yielding an exact eight-direction balanced set.
    if sector == 4:
        order = {"main": 0, "sector4_smoke": 1, "sector4_formal": 2}
        candidates.sort(key=lambda e: (order[e["source"]], e["demo"]))
    else:
        candidates = [e for e in candidates if e["source"] == "main"]
        candidates.sort(key=lambda e: e["demo"])
    if len(candidates) < 8:
        raise SystemExit(f"sector {sector} has only {len(candidates)} valid candidates")
    selected.extend(candidates[:8])

counts = {str(i): sum(e["sector"] == i for e in selected) for i in range(8)}
if counts != {str(i): 8 for i in range(8)}:
    raise SystemExit(f"not balanced: {counts}")

a.output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(a.output, "w") as out:
    out.attrs.update(format="factory_peg_insert_contact_recovery_native_v1_balanced64", selection="exactly eight strict native physical recovery trajectories per direction sector", source_files=json.dumps({name: str(path) for name, path in sources}))
    dst = out.create_group("demos")
    for i, entry in enumerate(selected):
        with h5py.File(entry["path"], "r") as src:
            src.copy(src["demos"][entry["demo"]], dst, name=f"demo_{i:05d}")
        g = dst[f"demo_{i:05d}"]
        g.attrs["source_file"] = entry["source"]
        g.attrs["source_demo"] = entry["demo"]
        entry["balanced_demo"] = f"demo_{i:05d}"

a.manifest.parent.mkdir(parents=True, exist_ok=True)
a.manifest.write_text(json.dumps({"benchmark": "contact_recovery_native_v1_balanced64", "direction_counts": counts, "entries": selected}, indent=2) + "\n")
print(json.dumps({"output": str(a.output), "manifest": str(a.manifest), "counts": counts}))
