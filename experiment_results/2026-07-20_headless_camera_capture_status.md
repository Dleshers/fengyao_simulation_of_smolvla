# Headless camera capture status (2026-07-20)

## Summary

Camera capture is still not fully working on the current AutoDL container. Physics-only/headless no-camera runs work, but true RGB camera observations are not available yet.

## What was verified

- Disk is sufficient: `/root/autodl-tmp` is 100G with ~55G free during this check.
- GPU is idle after cleanup: RTX 4080 SUPER, ~32G free, no Isaac camera probe left running.
- Official `isaaclab.python.headless.rendering.kit` with cameras still segfaults in `omni.kit.widget.viewport` during stage open.
- Installing and using Xvfb does not fix the segfault.
- A custom `isaaclab.python.headless.camera_minimal.kit` plus removing Replicator's hard dependency on `omni.kit.viewport.window` avoids the segfault and lets peg_insert env start.
- With that no-viewport-window route, RGB observations are empty: `rgb_camera.wrist_cam` shape is `(1, 0)`.
- A standalone cube + Replicator camera probe also returns empty RGB buffers for 10 rendered steps, so the issue is lower-level than peg_insert.
- `rep.orchestrator.step()` and manual `omni.usd.add_hydra_engine("rtx", ctx)` hang in this no-viewport route.

## Files changed/created

- Created custom kit:
  - `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.camera_minimal.kit`
- Patched IsaacLab `SimulationContext` to tolerate `get_active_viewport() is None`:
  - `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/source/isaaclab/isaaclab/sim/simulation_context.py`
  - backup: `.bak_20260720_viewport_none_guard`
- Patched peg_insert probe/record scripts to keep cameras RGB-only when enabled:
  - `experiment/peg_insert_headless_probe.py`
  - `experiment/record_peg_insert_oracle_demos.py`
  - backups: `.bak_20260720_rgb_only_camera`
- Temporarily patched Replicator extension dependency to comment out `omni.kit.viewport.window`:
  - `_runtime/.../omni.replicator.core-1.11.35.../config/extension.toml`
  - backup: `.bak_20260720_no_viewport_window`
- Added optional widget hydra skip patch, but this path hangs and is not sufficient:
  - `_runtime/.../omni.kit.widget.viewport-107.0.7.../omni/kit/widget/viewport/impl/texture.py`
  - backup: `.bak_20260720_skip_hydra`
- Created standalone diagnostic probe:
  - `experiment/replicator_min_camera_probe.py`

## Key logs

- `persistent/logs/peg_insert_camera_xvfb_rendering_rgb_only.log`: Xvfb + official rendering kit still segfaults.
- `persistent/logs/peg_insert_camera_minimal_rgb_only.log`: env starts but RGB shape is `(1, 0)`.
- `persistent/logs/replicator_min_camera_custom_no_viewport_window.log`: standalone cube camera also returns empty buffers.
- `persistent/logs/replicator_min_camera_custom_no_viewport_manual_hydra.log`: manual Hydra attach hangs after `hydra_before=[]`.

## Interpretation

The current blocker is the AutoDL/Isaac Sim rendering stack, not the peg_insert task code. The viewport-based path crashes; the no-viewport path cannot drive RTX/Hydra render output, so Replicator returns empty RGB buffers.

## Suggested next steps

1. Prefer changing the runtime graphics stack: try an Isaac Sim 4.5 officially supported driver branch (Kit log recommends 535.129.03) or an AutoDL image known to support Isaac Sim camera/headless EGL.
2. If changing driver/image is not possible, continue non-visual or tactile/state experiments on this machine; those already work.
3. If visual experiments are mandatory, collect/evaluate on a machine where Isaac Sim headless rendering passes a standalone Replicator camera probe before running peg_insert demos.
