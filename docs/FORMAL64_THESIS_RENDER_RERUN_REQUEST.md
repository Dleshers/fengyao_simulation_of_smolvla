# Formal64 Thesis Render Re-run Request

## Status and scope

This request **supersedes the illustrative-render portion** of
`FORMAL64_QUALITATIVE_ASSET_COLLECTION_PROMPT.md`. Retain the previously
uploaded 84 x 84 policy RGB images and traces as audit material, but do not use
them as the thesis illustrations.

The first collection successfully preserved the official Formal64 snapshot
provenance and outcome metadata, but its high-resolution overview views did
not clearly show the peg-in-hole contact region. The 84 x 84 policy-camera
images are valid raw model inputs, but are too low-resolution for a reader to
interpret recovery or ejection behaviour after enlargement.

This re-run must produce high-resolution, clean, visually interpretable
illustrative renders from the same frozen Formal64 snapshot replays. It must
not change the evaluation protocol or quantitative results.

Only these conditions are in scope: `visual`, `torque-original`, and
`torque-zero`. Do not produce shuffle-condition assets.

## Non-negotiable experimental constraints

1. Reuse the selected official snapshots, checkpoint, configuration, seed,
   controller, action scaling, termination criteria, and branch order from the
   first collection.
2. A paired comparison must use the exact same saved initial snapshot for each
   branch. Do not sample a new or easier state for a better picture.
3. Create dedicated **illustrative** cameras/render products only. Do not alter
   the existing policy cameras or their 84 x 84 RGB inputs.
4. Keep the official outcome label distinct from any replay observation. Include
   both in metadata when applicable.
5. Do not use GUI screenshots, debug primitives, oracle annotations, outcome
   labels, arrows, or edited imagery in the exported PNGs.

## Root cause to avoid

The earlier oblique overview mostly showed the ground grid, and the side
overview framed the robot without making the peg/hole interaction readable.
This must be treated as a failed framing check, not as an acceptable overview.
Before exporting, inspect each full-resolution PNG and adjust only the
dedicated illustrative camera pose/target until the acceptance checks below
pass.

## Camera and image requirements

Create one or more dedicated Replicator cameras after restoring the official
snapshot. They are for rendering only and must observe the replayed physical
state without modifying it.

- Preferred resolution: 1920 x 1080 PNG; absolute minimum: 1280 x 720 PNG.
- Use a stable fixed camera pose within each paired comparison.
- Frame the robot end effector, grasped peg, insertion hole, and nearby fixture
  together. The peg/hole contact region must be central and visually readable,
  rather than a small detail at the image edge.
- Use an oblique three-quarter contact view for the main illustration and a
  side/contact view when it helps reveal insertion depth or ejection.
- Maintain natural simulator materials and lighting. Do not replace assets,
  colours, or scene geometry for presentation.
- No visible UI, grid-dominant view, empty frame, severe clipping, or camera
  placed inside a robot link.

### Visual acceptance check for every exported frame

Confirm visually, before upload, that all of the following are true:

1. the robot wrist/end effector is visible;
2. the peg is visible;
3. the target hole/fixture opening is visible;
4. the peg-to-hole spatial relationship is visible without needing an overlay;
5. the image can be placed at approximately half an A4 page width and still be
   interpreted by a thesis reader; and
6. paired branch images use the same camera pose and crop.

If a terminal state no longer contains the peg/hole in frame, preserve the
fixed camera and provide an additional close contact frame immediately before
the state leaves the visible workspace. Record both frame indices in metadata.

## Required re-rendered assets

### A. Task overview

Render two clean full-resolution task views:

1. an oblique overview showing the robot, end effector, grasped peg, target
   hole/fixture, and local work surface; and
2. a closer side or oblique contact view in which the peg and hole are clearly
   distinguishable.

Use a representative official Formal64 snapshot (record its `pair_id` and
branch). Do not render an arbitrary reset state.

Suggested paths:

```text
formal64_thesis_rerender_20260819/task_overview/
  task_contact_overview_oblique.png
  task_contact_overview_side.png
  metadata.json
```

### B. Original-only-success paired cases

Re-render the existing audited pairs:

- `pair_id=14`: low load, sector 7; official visual failure and original-torque
  success;
- `pair_id=3`: high load, sector 1; official visual failure and original-torque
  success.

For each pair and for both `visual` and `torque-original`, export three frames
from one fixed illustrative contact camera:

1. `initial`: the restored near-contact state;
2. `recovery`: a frame showing the lateral correction/contact-stage recovery;
3. `terminal`: the final success/failure state, or the latest meaningful
   contact frame plus final state if the terminal state leaves the camera view.

The figure must make the physical difference between the branches interpretable
without altering the branch trajectory. The existing 84 x 84 camera 1/camera 2
files remain separate audit assets and should not be overwritten.

Suggested paths:

```text
formal64_thesis_rerender_20260819/paired_original_only_success/<pair_id>/
  visual_illustrative_initial.png
  visual_illustrative_recovery.png
  visual_illustrative_terminal.png
  torque_original_illustrative_initial.png
  torque_original_illustrative_recovery.png
  torque_original_illustrative_terminal.png
  metadata.json
```

### C. Original-torque ejection safety case

Re-render `pair_id=42` (low load, sector 5): the frozen official label is
original-torque failure with ejection, while visual and zero torque are official
non-ejection successes.

From the same illustrative contact camera, export:

1. original torque at `initial`;
2. original torque `pre_ejection`, immediately before the ejection event;
3. original torque `ejection_or_terminal`, with the event visually readable;
4. a matched `torque-zero` terminal/comparator frame; and
5. optionally, a matched visual comparator frame if it adds clarity.

Suggested paths:

```text
formal64_thesis_rerender_20260819/original_ejection_safety_case/42/
  torque_original_illustrative_initial.png
  torque_original_illustrative_pre_ejection.png
  torque_original_illustrative_ejection_or_terminal.png
  torque_zero_illustrative_terminal.png
  metadata.json
```

## Metadata and audit requirements

Write a `metadata.json` for each case and include at least:

```json
{
  "pair_id": 3,
  "case_type": "paired_original_only_success",
  "branch": "visual",
  "official_strict_success": false,
  "official_ejection": false,
  "replay_outcome": "...",
  "load_band": "high",
  "direction_sector": 1,
  "checkpoint": "...",
  "seed": "...",
  "snapshot_sha256": "...",
  "illustrative_camera_pose": "...",
  "illustrative_camera_target": "...",
  "resolution": [1920, 1080],
  "frames": {
    "initial": 0,
    "recovery": 0,
    "terminal": 0
  }
}
```

Also provide a short `ASSET_COLLECTION_REPORT.md` that maps every PNG to its
`pair_id`, branch, frame, official outcome, resolution, and absolute server
path. State whether every image passed the six-point visual acceptance check.

## Headless implementation notes

Headless capture is expected to work through the existing Isaac Lab/Isaac Sim
launcher with camera rendering enabled. Use the server's supported headless
flags and graphics backend. Call the simulation render step before reading the
dedicated illustrative render-product RGB output. The dedicated high-resolution
camera must be created independently of the policy observation cameras.

## Upload requirement

After validation, add the new directory under:

```text
experiment_results/formal64_thesis_rerender_20260819/
```

Commit the PNGs, metadata, manifest/report, and any small supporting files to
the repository's `main` branch and push to the configured GitHub remote. Do not
delete or overwrite the existing
`experiment_results/formal64_qualitative_assets_20260819/` audit directory.
