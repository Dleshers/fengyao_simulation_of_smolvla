# Pixel and continuity review

The renderer-level checks passed: all four files are RGB PNGs at 1920×1080,
non-uniform, and generated with one fixed camera and lighting setup. The
attachment check is structural: the identical USD translation is applied to
the complete `Robot` and `HeldAsset` roots for every panel.

The `visual_review` column remains `pending_user_visual_check` because the
remote machine has no GUI/UDP path for an interactive human viewport. The
review sheet is provided so the four panels can be inspected locally before
thesis insertion.

| image | wrist_visible | peg_visible | peg_attached_to_wrist | fixture_visible | hole_visible | relationship_visible | same_camera | visual_review |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `01_initial_lateral_error.png` | true | true | true | true | true | true | true | pending_user_visual_check |
| `02_contact_misalignment.png` | true | true | true | true | true | true | true | pending_user_visual_check |
| `03_lateral_recovery.png` | true | true | true | true | true | true | true | pending_user_visual_check |
| `04_aligned_insertion.png` | true | true | true | true | true | true | true | pending_user_visual_check |

These are conceptual scripted states, not Formal64 policy outputs or measured
success/failure examples.

