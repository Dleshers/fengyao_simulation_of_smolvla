# Formal64 Thesis Render Correction Request

## Blocking status

Do **not** treat the assets in
`experiment_results/formal64_thesis_rerender_20260819/` as thesis-ready. They
must not be inserted into the report in their current form.

The local review found two blocking defects:

1. **The rendered pixels fail the intended framing requirement.** Although the
   metadata reports `acceptance=true`, several 1920 x 1080 images show mainly a
   ground grid, empty background, or a workbench edge. The wrist, peg, hole,
   and their contact relationship are not simultaneously readable.
2. **Some replay trajectories do not match the official Formal64 outcome that
   the planned figure would claim.** In particular:
   - pair 3: `torque-original` is an official strict success, but the supplied
     replay metadata says `success=false`;
   - pair 42: `torque-original` is an official ejection failure, but the
     supplied replay metadata says `success=true, ejection=false`.

The reported Formal64 aggregate statistics remain unchanged. This request is
only for valid qualitative illustrative material with auditable provenance.

## Absolute integrity rule

An image may be captioned as an official success, visual failure, or ejection
case only if it was rendered from the corresponding **official recorded
trajectory/state sequence** or a deterministically verified replay with the
same outcome. A fresh replay that reaches a different outcome is not an
acceptable substitute.

Do not alter official labels to agree with a divergent replay, and do not use a
divergent replay image with the official label. If the official trajectory/state
sequence cannot be recovered for a requested case, report it as unavailable and
do not fabricate a qualitative replacement.

## Required diagnostic work before rendering

For pair IDs 3, 14, and 42, identify the exact source of the official outcome
in the frozen Formal64 artefacts. Determine whether a complete state/action
trajectory, per-step trace, or recoverable deterministic trajectory exists.

For every branch to be illustrated, record:

- `pair_id`, condition, official strict-success/ejection label;
- source file and trajectory/episode identifier for the official path;
- snapshot hash, checkpoint, seed, action sequence, and controller settings;
- whether the render came from the official path or a verified replay;
- if replayed, the exact verification that its strict-success/ejection outcome
  equals the official label.

If the first replay outcome diverges, diagnose the divergence before rendering:
for example, un-restored controller state, action RNG, physics state,
randomisation state, or an incorrect trajectory/action sequence. Do not simply
try another seed until an image looks suitable.

## Pixel-level framing correction

The existing geometry-distance checks are insufficient. Before upload, a person
or vision-capable agent must inspect the actual PNG pixels at normal display
size. A frame passes only when all conditions below are visibly true:

1. wrist/end effector, grasped peg, target hole/fixture opening, and local
   contact area are all in the same image;
2. the peg--hole relationship is central and large enough to understand when
   the image is placed at roughly half an A4 page width;
3. no camera is inside a mesh, clipped by the workbench, pointed at empty
   space, or dominated by the ground grid;
4. paired branches use the identical illustrative-camera pose, target, lens,
   clipping range, lighting, and crop; and
5. the PNG has no GUI, debug overlay, oracle annotation, outcome label, or
   manually edited content.

The previous high-resolution image size is acceptable, but resolution alone is
not evidence of a usable figure. Use 1920 x 1080 PNG where possible, with
dedicated illustrative render products entirely separate from the 84 x 84
policy input cameras.

## Required corrected assets

### 1. Task illustration

From an official Formal64 near-contact state, provide:

- `task_contact_overview_oblique.png`: a clear three-quarter view of end
  effector, peg, fixture/hole, and work surface;
- `task_contact_overview_side.png`: a close view that makes insertion depth and
  the peg/hole opening readable.

If a full robot view and a contact close-up cannot be achieved with one camera,
provide both as separate fixed illustrative views. Neither image may be an
empty-grid or workbench-only view.

### 2. Pair 14 low-load original-only success

Use the official low-load, sector-7 pair 14 trajectory:

- `visual`: official strict failure;
- `torque-original`: official strict success.

For each branch, export `initial`, `recovery`, and `terminal` frames from one
fixed contact camera. The terminal frame must visibly support the caption, or a
documented pre-terminal close-contact frame must be included in addition.

### 3. Pair 3 high-load original-only success

Use the official high-load, sector-1 pair 3 trajectory:

- `visual`: official strict failure;
- `torque-original`: official strict success.

The prior torque-original replay failed and is invalid for a figure claiming
the official success. First recover or deterministically reproduce the official
successful path; otherwise omit pair 3 from the deliverable and state why.

### 4. Pair 42 original-torque ejection safety case

Use the official low-load, sector-5 pair 42 trajectory:

- `torque-original`: official strict failure with ejection;
- `torque-zero`: official strict success without ejection.

Export original-torque `initial`, `pre_ejection`, and `ejection_or_terminal`,
plus the zero-torque terminal comparator. The prior original-torque replay did
not eject and is invalid for an ejection figure.

## Mandatory visual review evidence

In addition to individual clean PNGs, produce an unedited `REVIEW_CONTACT_SHEET.png`
that places all proposed thesis images at approximately their intended printed
size. This contact sheet is only for review and may use minimal filenames in
the margins; it will not be inserted in the thesis.

Provide `PIXEL_REVIEW.md` with one row per image:

| file | pair ID | branch | official outcome | rendered-path outcome | reviewer | pass/fail | reason |
|---|---:|---|---|---|---|---|---|

Every proposed thesis image must be marked `pass`. If an asset fails visual
review or outcome verification, retain it only as diagnostic material outside
the proposed thesis-image directory.

## Metadata and upload

Write a complete `MANIFEST.json` and `ASSET_COLLECTION_REPORT.md` stating image
resolution, camera pose/target, outcome provenance, and the exact absolute
server source path. Keep the original asset directories unchanged.

Upload corrected, approved assets under:

```text
experiment_results/formal64_thesis_render_corrected_20260819/
```

Commit this new directory, including PNGs, metadata, `PIXEL_REVIEW.md`, and the
review contact sheet, to the repository `main` branch and push to GitHub. Do
not overwrite `formal64_qualitative_assets_20260819` or
`formal64_thesis_rerender_20260819`.
