# Pixel review

Both raw images are 1920×1080 RGB PNGs rendered with one camera and one light
setup. The actual Panda joint state is changed for the misaligned panel; the
held peg is translated by the measured fingertip displacement so the grasp
remains visually coherent.

| image | intended state | same camera | wrist/end effector | measured lateral displacement | review status |
|---|---|---:|---:|---:|---|
| `02_contact_misalignment.png` | near-rim contact, not a coarse offset | true | required | 8.33 mm | renderer checks passed; local visual check recommended |
| `03_lateral_recovery.png` | recovered/aligned contact state | true | required | 0 mm | renderer checks passed; local visual check recommended |
