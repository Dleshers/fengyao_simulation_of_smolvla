# Same-camera lateral-recovery pair (joint-motion revision)

This is the revised pair for visually demonstrating lateral recovery. The
previous USD-root translation was largely overwritten by Isaac Sim's physics
synchronization and produced only a subtle image difference. This revision
changes the actual Panda joint-1 state for the misaligned panel, measures the
resulting fingertip displacement, and moves the held peg by that measured
displacement so the grasp remains visually coherent.

- `02_contact_misalignment.png`: joint 1 offset by `+0.18 rad`; measured
  fingertip displacement is approximately `(-9.9 mm, +99.4 mm, 0)`.
- `03_lateral_recovery.png`: the recorded joint state is restored and the
  measured displacement returns to zero.

Both panels use the same camera and lighting. They are scripted conceptual
illustrations, not Formal64 evaluation trajectories, policy outputs, or
quantitative evidence. Place them side by side; the lateral change is intended
to be directly visible without an overlay.
