# Pixel review

Both raw PNGs are 1920×1080 RGB images rendered with one camera and one light
setup. The Robot and HeldAsset roots receive the same USD translation, so the
peg remains attached to the wrist in both panels.

| image | intended state | same camera | wrist/end effector required | peg/hole relation | review status |
|---|---|---:|---:|---:|---|
| `02_contact_misalignment.png` | larger lateral error | true | true | visible | renderer checks passed; local visual check recommended |
| `03_lateral_recovery.png` | reduced lateral error | true | true | visible | renderer checks passed; local visual check recommended |
