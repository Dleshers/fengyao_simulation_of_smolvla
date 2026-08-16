# High-load tactile evaluation protocol (2026-08-17)

## Status of the balanced 64-pair evaluation

- Completed controlled-contact comparison: 8 directions x 2 load bands x 4 same-snapshot pairs per cell.
- Every pair passed state, RGB, and torque-history identity auditing.
- Visual strict success: 52/64 (81.2%).
- Original-torque strict success: 55/64 (85.9%).
- Overall difference: +4.7 percentage points; the all-load preregistered gate is **FAIL**.
- Low load: visual 30/32 (93.8%), torque 27/32 (84.4%).
- High load: visual 22/32 (68.8%), torque 28/32 (87.5%), a +18.75-point difference.
- In high load, torque had 8 unique paired wins versus 2 for visual; exact paired-test p is about 0.109.

## Interpretation boundary

The completed result does not support a claim that torque improves insertion across all loads: it misses the global +10-point gate, is negative at low load, and does not meet aggregate safety non-inferiority.

The high-load stratum is a positive, physically motivated observation. The permitted wording is: *Under high-load near-hole physical contact, causal original torque information showed a positive recovery trend relative to vision-only control.* It is not yet a statistically confirmed general effect.

## Independent high-load confirmation

- Purpose: test the high-load hypothesis independently, without pooling its result with the exploratory 32-pair subgroup.
- Frozen policies: hard80 10k visual and original-torque arms; same dataset, action clips, inference settings, and 30x7 causal torque history.
- New evaluation: 64 high-load-only controlled-contact pairs.
- Stratification: 8 offset directions, exactly 8 pairs per direction; no reuse of the balanced-run snapshots.
- Seed: `20270817`; all branches restore the same physical snapshot and must pass the existing state/RGB/torque-hash audit.
- Gate: torque success gain >=10 points, positive paired net wins, non-inferior in >=6/8 directions, and no increased ejection, pass-through, or grasp drift.
- Results will be reported separately; only an independent pass supports a high-load tactile benefit conclusion.

## Artifacts

- Balanced result: `persistent/evaluation_results/visual_torque_4090_master_20260816/baseline_hard80_10k_formal64_resume_attempt1/`.
