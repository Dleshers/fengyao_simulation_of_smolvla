# Simulator-Rendered Conceptual Illustration Request

## Purpose

Create a small set of clean, simulator-rendered **conceptual illustrations**
for the thesis explanation of contact-stage error recovery. These images are
not replacements for the audited Formal64 results and must not be presented as
paired evaluation outcomes.

The figures may be used in the Methodology or Discussion chapters to explain
the geometry and the intended mechanism:

1. visual coarse localisation leaves a lateral peg--hole error;
2. contact is established and the state becomes sensitive to lateral error;
3. a torque-history-conditioned controller can produce a corrective lateral
   motion; and
4. the peg becomes aligned with the hole and insertion can continue.

## Critical interpretation boundary

These are scripted illustrative configurations, not Formal64 trajectories.
They must not be used to claim that visual control failed, torque-original
succeeded, or a particular branch ejected. Do not label any panel as a measured
success/failure case, pair ID, or statistical result.

Every thesis caption must contain wording equivalent to:

> Simulator-rendered conceptual illustration of contact-stage recovery. The
> states are scripted illustrative configurations and are not Formal64
> evaluation trajectories or quantitative outcomes.

The existing Formal64 quantitative tables and paired-outcome figure remain the
only evidence for measured success rates, McNemar tests, ejections, and formal
decisions.

## Rendering procedure

Use Isaac Sim/Isaac Lab in the supported headless mode with the normal Factory
peg-in-hole assets, lighting, materials, robot, peg, hole, and fixture. The
illustrations should be generated from a deterministic scripted scene sequence,
not from a divergent Formal64 replay.

1. Reset a clean simulation scene with the robot, grasped peg, target fixture,
   and hole visible.
2. Set a representative near-contact lateral offset using a documented
   kinematic waypoint or controlled initial pose.
3. Render the following four states with one fixed illustrative camera:
   - `01_initial_lateral_error`: peg visibly offset from the hole;
   - `02_contact_misalignment`: peg is near/contacting the fixture while still
     laterally misaligned;
   - `03_lateral_recovery`: peg has moved laterally towards alignment;
   - `04_aligned_insertion`: peg is visually aligned and entering the hole.
4. The lateral recovery and insertion motions may be scripted waypoints or a
   simple controller used only to construct the explanatory scene. They must
   not be described as policy outputs or evaluation outcomes.
5. Save the scene/waypoint configuration and the exact frame or simulation step
   for every rendered state.

## Camera and image requirements

- Render lossless PNG images at 1920 x 1080 where possible; 1280 x 720 is the
  minimum.
- Use one stable three-quarter contact camera for the four-panel sequence. A
  second side/contact camera may be provided if it makes insertion depth easier
  to understand.
- The wrist/end effector, grasped peg, target hole/fixture opening, and local
  contact region must be visible in every panel.
- The peg--hole relationship must be central and readable at approximately half
  an A4 page width.
- Avoid a ground-grid-dominant view, empty background, severe clipping, camera
  placement inside a mesh, or a workbench edge that hides the contact area.
- Do not add GUI chrome, debug geometry, oracle state, success/failure labels,
  arrows, or manually edited pixels to the raw PNGs. Labels can be added in the
  LaTeX figure layout or a separate vector diagram.
- This is an illustrative render product, not a policy RGB input. Do not
  overwrite or replace the existing 84 x 84 policy observation assets.

## Suggested deliverables

```text
experiment_results/simulator_conceptual_illustrations_20260819/
  contact_recovery_sequence/
    01_initial_lateral_error.png
    02_contact_misalignment.png
    03_lateral_recovery.png
    04_aligned_insertion.png
    metadata.json
  contact_recovery_sequence_side_view/
    01_initial_lateral_error.png
    02_contact_misalignment.png
    03_lateral_recovery.png
    04_aligned_insertion.png
    metadata.json
  MANIFEST.json
  README.md
```

The side-view sequence is optional. The principal deliverable is one coherent
four-panel sequence with an identical camera pose, crop, lighting, and scene
geometry across panels.

## Required metadata

The metadata must state explicitly that the sequence is conceptual and
scripted. Include at least:

```json
{
  "case_type": "simulator_rendered_conceptual_illustration",
  "formal64_pair_id": null,
  "is_formal64_evaluation_trajectory": false,
  "is_policy_output": false,
  "scene_seed": "...",
  "script_or_waypoint_config": "...",
  "camera_pose": "...",
  "camera_target": "...",
  "resolution": [1920, 1080],
  "states": {
    "01_initial_lateral_error": {"step": 0, "png": "..."},
    "02_contact_misalignment": {"step": 0, "png": "..."},
    "03_lateral_recovery": {"step": 0, "png": "..."},
    "04_aligned_insertion": {"step": 0, "png": "..."}
  }
}
```

Do not insert a Formal64 pair ID or official outcome into this metadata. If the
scene happens to start from a Formal64 snapshot, record that provenance only as
source context; the scripted sequence still remains conceptual and must not be
interpreted as the corresponding policy branch.

## Review checklist

Before upload, inspect the actual pixels at normal display size and confirm:

- peg and hole are visible in every panel;
- the lateral offset, contact, recovery direction, and aligned insertion are
  visually distinguishable;
- all four panels use the same camera and crop;
- there are no overlays or misleading outcome annotations; and
- the README and caption explicitly state that the sequence is conceptual,
  scripted, and not a Formal64 result.

If any panel fails these checks, adjust the illustrative camera or scripted
scene and re-render it. Do not mark a failed render as acceptable merely because
its resolution or object-distance metadata is valid.

## Upload

After visual review, commit the new directory and its metadata to the
repository's `main` branch and push it to the configured GitHub remote. Do not
modify or delete the existing Formal64 quantitative or audit directories.
