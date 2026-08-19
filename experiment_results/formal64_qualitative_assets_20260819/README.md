# Formal64 qualitative assets (2026-08-19)

This directory contains the qualitative assets collected from the frozen Formal64 evaluation snapshots after following `docs/FORMAL64_QUALITATIVE_ASSET_COLLECTION_PROMPT.md`.

## Contents

- `task_overview/`: oblique and side 1280×720 task views.
- `paired_original_only_success/14/`: low-load paired case where original torque was the official success and visual was not.
- `paired_original_only_success/3/`: high-load paired case where original torque was the official success and visual was not.
- `original_ejection_safety_case/42/`: ejection-safety case with visual/zero comparators.
- Each condition directory contains initial, recovery, and terminal camera 1/2 PNGs, `trace.npz`, and `metadata.json`.
- `MANIFEST.json`: selected case IDs, checkpoint provenance, snapshot hashes, and asset mapping.
- `ASSET_COLLECTION_REPORT.md`: generated inventory report.

## Conditions

Only `visual`, `torque-original`, and `torque-zero` are included. The shuffle condition was intentionally not collected.

All branches were replayed from the corresponding official Formal64 CPU snapshot. The frozen official outcome labels remain separate from replay outcomes in the metadata; therefore these assets are suitable for qualitative side-by-side figures without silently changing the quantitative evaluation protocol.

## Provenance

- Official result: `experiment_results/HIGH_LOAD_CONFIRMATORY_20260817_RESULTS.json` and the Formal64 evaluation artifacts referenced by `MANIFEST.json`.
- Prompt: `docs/FORMAL64_QUALITATIVE_ASSET_COLLECTION_PROMPT.md`.
- The PNGs are rendered without GUI/debug overlays. The NPZ traces include the 30×7 torque-history window, executed actions, pose/error telemetry, and safety flags.
