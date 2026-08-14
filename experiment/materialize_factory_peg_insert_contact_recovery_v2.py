#!/usr/bin/env python3
"""Materialize one auditable shared-data corpus from balanced64 plus hard failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-input", type=Path, required=True)
    parser.add_argument("--failure-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-balanced", type=int, default=64)
    parser.add_argument("--expected-failures", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        ("balanced_native_contact", args.balanced_input, args.expected_balanced, (
            "factory_peg_insert_contact_recovery_native_v1",
            "factory_peg_insert_contact_recovery_native_v1_balanced64",
        )),
        ("actual_policy_submm_failure", args.failure_input, args.expected_failures, (
            "factory_peg_insert_policy_failure_recovery_v1",
        )),
    ]
    rows = []
    with h5py.File(partial, "w") as target:
        target.attrs.update(
            format="factory_peg_insert_contact_recovery_v2_actual_failure80",
            composition="64 balanced native-contact recovery + 16 actual hybrid-policy sub-mm failure recovery",
            training_contract="visual and torque arms must use this identical corpus, seed, steps, and action loss",
            evaluation_contract="paired unseen seeds; never use oracle pose metadata as policy input",
        )
        out_demos = target.create_group("demos")
        out_index = 0
        for source_name, source_path, expected, allowed_formats in sources:
            with h5py.File(source_path, "r") as source:
                raw_format = str(source.attrs.get("format", ""))
                if raw_format not in allowed_formats:
                    raise ValueError(f"{source_path}: unexpected format={raw_format!r}")
                demos = sorted(source["demos"])
                if len(demos) != expected:
                    raise ValueError(f"{source_path}: demos={len(demos)}, expected={expected}")
                for source_demo in demos:
                    output_demo = f"demo_{out_index:05d}"
                    source.copy(source["demos"][source_demo], out_demos, name=output_demo)
                    group = out_demos[output_demo]
                    group.attrs["origin_dataset"] = source_name
                    group.attrs["origin_demo"] = source_demo
                    rows.append({
                        "demo": output_demo,
                        "origin_dataset": source_name,
                        "origin_demo": source_demo,
                        "pair_id": str(group.attrs.get("pair_id", "")),
                        "episode_seed": int(group.attrs.get("episode_seed", -1)),
                        "direction_sector": int(group.attrs.get("direction_sector", -1)),
                        "load_band": int(group.attrs.get("load_band", -1)),
                        "frames": len(group["state"]),
                        "recovery_label_frames": int(group.attrs.get("recovery_label_frames", 0)),
                    })
                    out_index += 1
        target.flush()
    os.replace(partial, args.output)

    manifest = {
        "format": "factory_peg_insert_contact_recovery_v2_actual_failure80",
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "sources": {
            source_name: {
                "path": str(source_path),
                "sha256": sha256(source_path),
                "demos": expected,
            }
            for source_name, source_path, expected, _ in sources
        },
        "total_demos": len(rows),
        "total_frames": sum(row["frames"] for row in rows),
        "total_recovery_label_frames": sum(row["recovery_label_frames"] for row in rows),
        "rows": rows,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": manifest["output_sha256"],
        "demos": manifest["total_demos"],
        "recovery_labels": manifest["total_recovery_label_frames"],
    }))


if __name__ == "__main__":
    main()
