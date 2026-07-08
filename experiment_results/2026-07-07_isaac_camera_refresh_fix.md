# Isaac camera refresh fix and corrected baseline diagnostic

Date: 2026-07-07

## Root cause and fix

`scripts/eval_server.py` uniquely called `parse_env_cfg(..., use_fabric=False)`.
The collection and replay paths leave the task's Fabric setting at its default.
In headless evaluation, the forced non-Fabric configuration allowed PhysX tensor
state to advance while RTX camera renders retained reset-time articulation and
object geometry.  Small frame-to-frame pixel changes were rendering noise, not
scene motion.

The fix removes the `use_fabric=False` override and restores the task default:

```python
env_cfg = parse_env_cfg(self.env_name, device=self.device, num_envs=1)
```

## Deterministic camera freshness smoke test

`remote_handoff_gripper_lstm_work/experiment/probe_camera_refresh.py` executes a
smooth 60-step joint-space trajectory without loading a policy and compares the
first and last server RGB observations.

Result:

- joint-position change (L2): `0.87973 rad`;
- table-camera first/last MAE: `11.80393` (`14,222` pixels changed by >10);
- wrist-camera first/last MAE: `23.37714` (`22,112` pixels changed by >10);
- visual inspection confirms that robot/cube geometry changes in both cameras.

Artifacts:

- `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/camera_refresh_fabric_smoke_20260707/result.json`
- the adjacent `table_first.png`, `table_last.png`, `wrist_first.png`, and
  `wrist_last.png` files.

## Corrected one-episode baseline result

Configuration:

- baseline `smolvla-franka-pickplace-baseline-50k-seed1000`;
- seed `1002`, `n_action_steps=1`;
- fixed cube `[0.425, -0.115, 0.022]`;
- no gripper gate;
- live Fabric-synchronized table and wrist RGB.

Result:

- reported success: `false` (episode ended at step 164);
- first close command: step `53`, EEF-cube distance `0.05104 m`;
- minimum EEF-cube distance before terminal auto-reset: `0.03576 m`;
- cube maximum height: `0.21186 m` (initial `0.02200 m`);
- maximum cube displacement: `0.29422 m`;
- the video shows a real grasp, lift, transport, and placement attempt at the
  basket rim.

Artifacts:

- `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_camera_refresh_fixedcube_seed1002_20260707/`

## Impact on earlier conclusions

Yes, stale RGB directly affected visual decision-making.  SmolVLA repeatedly
received reset-time table/wrist geometry while its 9D joint/gripper state kept
changing.  It therefore could not observe approach progress, contact, cube
motion, grasp success, or basket-relative placement.  The dramatic behavioral
change after restoring live images demonstrates causality: the same baseline
progressed from push/no-lift behavior to grasp-lift-transport-near-placement.

All earlier baseline visual evaluations produced by the forced non-Fabric
server must be marked invalid and rerun.  Their physics telemetry can describe
the executed trajectories, but cannot measure normal visual closed-loop policy
quality.

## Remaining issue

The corrected run reaches the basket but is still reported unsuccessful.  On
the terminal server step, the manager-based environment auto-resets before the
wrapper extracts observation and calls `_check_success()`.  The logged terminal
`*_after` state is consequently the next reset state, not the true terminal
state.  Success/termination capture must be moved to terminal information or
recorded before auto-reset before interpreting the final placement failure.

