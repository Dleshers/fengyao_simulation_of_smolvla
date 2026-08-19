# Pixel and continuity review

The rows below refer to the raw PNGs, not the labeled contact sheet. Boolean fields are explicit review claims for the corrected fixed-camera sequence.

| image | wrist_visible | peg_visible | peg_attached_to_wrist | fixture_visible | hole_visible | peg_hole_relationship_visible | same_camera_as_sequence |
|---|---:|---:|---:|---:|---:|---:|---:|
| `01_initial_lateral_error.png` | true | true | true | true | true | true | true |
| `02_contact_misalignment.png` | true | true | true | true | true | true | true |
| `03_lateral_recovery.png` | true | true | true | true | true | true | true |
| `04_aligned_insertion.png` | true | true | true | true | true | true | true |

Additional checks:

- Every raw image is a 1920×1080 RGB PNG and has non-uniform simulator pixels; `render_validation.json` records means, standard deviations, and frame differences.
- The same camera pose, target, clipping range, crop, lens, and lighting are used for all four frames.
- The robot root and held-peg root receive the same waypoint translation; the attachment vector is recorded in `metadata.json`.
- The four panels are scripted explanatory configurations only; no row is a Formal64 trajectory, policy output, or measured result.
