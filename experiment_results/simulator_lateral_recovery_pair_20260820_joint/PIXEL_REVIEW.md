# Pixel review

Both raw images are 1920×1080 RGB PNGs rendered with one camera and one light
setup. The joint-motion renderer measured the fingertip displacement rather
than relying on a USD root translation. The held peg is translated by the same
measured displacement in each panel.

| image | intended state | same camera | wrist/end effector | measured lateral displacement | review status |
|---|---|---:|---:|---:|---|
| `02_contact_misalignment.png` | joint-1 offset / visible lateral error | true | required | about 99.4 mm in y | renderer checks passed; local visual check recommended |
| `03_lateral_recovery.png` | restored joint state / recovery | true | required | 0 mm relative to baseline | renderer checks passed; local visual check recommended |
