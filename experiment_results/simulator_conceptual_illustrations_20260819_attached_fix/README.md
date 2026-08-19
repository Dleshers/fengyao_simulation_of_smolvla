# Attached-rigid conceptual contact-recovery illustrations

This revision corrects the previous third-panel defect: the Factory robot has
a fixed base, so changing its articulation root pose did not move the rendered
wrist. The earlier renderer therefore moved the held peg without moving the
wrist and could show a floating peg. The new renderer applies the same USD
translation to `/World/envs/env_0/Robot` and `/World/envs/env_0/HeldAsset`,
preserving the visual wrist-to-peg relationship in every panel. The dedicated
camera was also widened and moved back so the wrist remains in frame during
recovery and aligned insertion.

The four PNGs are scripted simulator illustrations only:

> Simulator-rendered conceptual illustration of contact-stage recovery. The
> states are scripted illustrative configurations and are not Formal64
> evaluation trajectories or quantitative outcomes.

The raw images contain no GUI, debug geometry, oracle state, outcome label, or
manually edited pixels. `REVIEW_CONTACT_SHEET.png` is for visual review only.

## Files

- `contact_recovery_sequence/`: four 1920×1080 lossless PNGs and metadata.
- `REVIEW_CONTACT_SHEET.png`: 2×2 review sheet, not a thesis raw asset.
- `PIXEL_REVIEW.md`: explicit continuity checks and review status.
- `render_validation.json`: non-blank/render-shape checks.
- `MANIFEST.json`: hashes, camera, script and provenance.

