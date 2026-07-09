#!/usr/bin/env python3
"""Audit eval_info.json vs trajectory.jsonl success bookkeeping.

The LeRobot eval summary should be treated as authoritative for final success
rate.  The Isaac trajectory log contains richer per-step diagnostics, but
termination-manager terms can disagree with the reported `is_success` on
timeout frames.  This helper makes that disagreement explicit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_eval_successes(path: Path) -> list[bool]:
    info = json.loads(path.read_text())
    return [bool(x) for x in info["per_task"][0]["metrics"]["successes"]]


def load_final_steps(path: Path) -> dict[int, dict]:
    by_episode: dict[int, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "step":
                by_episode[int(row["episode"])].append(row)
    return {episode: rows[-1] for episode, rows in by_episode.items() if rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-info", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    args = parser.parse_args()

    successes = load_eval_successes(args.eval_info)
    finals = load_final_steps(args.trajectory)

    print("episode,eval_info_success,jsonl_is_success,termination_success,hit_max_steps,env_step_done,disagree")
    for episode, eval_success in enumerate(successes):
        final = finals.get(episode, {})
        term_success = final.get("termination_terms", {}).get("success")
        jsonl_success = final.get("is_success")
        disagree = (
            term_success is not None
            and jsonl_success is not None
            and bool(term_success) != bool(jsonl_success)
        )
        print(
            f"{episode},{eval_success},{jsonl_success},{term_success},"
            f"{final.get('hit_max_steps')},{final.get('env_step_done')},{disagree}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
