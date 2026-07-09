#!/usr/bin/env python3
"""Merge local LeRobot datasets without contacting Hugging Face Hub.

The public `lerobot-edit-dataset --operation.type merge` command accepts a
single root argument, but this workspace stores datasets as
`<dataset_parent>/<repo_id>`.  This helper passes explicit source roots to the
lower-level aggregate function so all reads stay local and the official source
dataset is not modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> None:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parent", required=True, type=Path)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--repo-ids", nargs="+", required=True)
    parser.add_argument(
        "--source-roots",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit roots matching --repo-ids. Defaults to <dataset-parent>/<repo_id>.",
    )
    parser.add_argument("--data-files-size-in-mb", type=float, default=None)
    args = parser.parse_args()

    parent = args.dataset_parent.resolve()
    output_root = parent / args.output_repo_id
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing merged dataset: {output_root}")

    if args.source_roots is not None and len(args.source_roots) != len(args.repo_ids):
        raise ValueError("--source-roots must have the same length as --repo-ids")

    roots = []
    for index, repo_id in enumerate(args.repo_ids):
        root = args.source_roots[index].resolve() if args.source_roots is not None else parent / repo_id
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Missing LeRobot dataset metadata for {repo_id}: {root}")
        roots.append(root)

    aggregate_datasets(
        repo_ids=args.repo_ids,
        roots=roots,
        aggr_repo_id=args.output_repo_id,
        aggr_root=output_root,
        data_files_size_in_mb=args.data_files_size_in_mb,
    )

    merged = LeRobotDataset(args.output_repo_id, root=output_root)
    summary = {
        "output_repo_id": args.output_repo_id,
        "output_root": str(output_root),
        "source_repo_ids": args.repo_ids,
        "total_episodes": merged.num_episodes,
        "total_frames": len(merged),
        "fps": merged.fps,
        "features": sorted(merged.features.keys()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
