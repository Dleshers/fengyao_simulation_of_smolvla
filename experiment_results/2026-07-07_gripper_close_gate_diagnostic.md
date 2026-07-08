# 2026-07-07 baseline gripper close gate diagnostic

## Purpose

This local-only diagnostic tested whether the pure SmolVLA baseline fails mainly because it closes the gripper too early.

The eval server was instrumented with a diagnostic `GRIPPER_CLOSE_GATE_EEF_CUBE_DIST` override: when the policy emits a negative gripper command, the server keeps the gripper open unless the measured EEF-cube distance is below the threshold. This does not retrain or overwrite any model/data artifact.

## Runs

Baseline policy:

`_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`

Common eval settings:

- seed: `1002`
- control mode: `joint`
- fixed cube pose: `0.425,-0.115,0.022`
- episodes: `1`
- policy: `n_action_steps=1`
- input mapping: `rgb_table -> camera1`, `rgb_wrist -> camera2`
- torque input: disabled for pure visual baseline

Outputs:

- no-gate fixed cube telemetry:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_center_effective_seed1002_20260707/baseline_eef_trajectory.jsonl`
- gate 0.050 m telemetry:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_gate005_seed1002_20260707/baseline_eef_trajectory.jsonl`
- gate 0.065 m telemetry:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_gate0065_seed1002_20260707/baseline_eef_trajectory.jsonl`
- gate 0.065 m video:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_gate0065_seed1002_20260707/videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`
- machine-readable comparison:
  `experiment_results/2026-07-07_gripper_close_gate_comparison.txt`

## Key results

| condition | success | first raw close | first effective close | min EEF-cube dist | cube max displacement | cube max lift | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| original random cube, no gate | false | step 63 @ 11.7 cm | step 63 | 11.50 cm | ~0.0 mm | ~0.0 mm | gripper closes far from cube; no contact |
| fixed cube, no gate | false | step 63 @ 7.0 cm | step 63 | 6.06 cm | 49.9 mm | 0.15 mm | contact improves but mainly pushes/slides cube |
| fixed cube, gate 0.050 m | false | step 61 @ 7.25 cm | never | 6.52 cm | 4.9 mm | 0.44 mm | arm never reaches 5 cm threshold; gripper remains open |
| fixed cube, gate 0.065 m | false | step 63 @ 7.05 cm | step 72 @ 6.31 cm | 5.52 cm | 38.2 mm | 4.66 mm | delayed close improves approach/lift slightly, but grasp remains unstable |

## Interpretation

The gate tests refine the failure mode:

1. Premature gripper closing contributes to failure, but is not sufficient as the sole explanation.
2. With a strict 5 cm gate, the arm never reaches the close threshold; this shows the visual baseline's arm approach plateaus offset from the cube.
3. With a 6.5 cm gate, closing is delayed by about 9 steps compared with the no-gate run. This improves minimum EEF-cube distance from 6.06 cm to 5.52 cm and produces a small transient cube lift, but still no successful grasp.
4. The dominant remaining issue is likely grasp geometry/contact alignment: the policy approaches from an offset pose, closes near the cube, nudges it laterally, and fails to trap/lift it stably.

This supports the current working hypothesis: the dataset/action labels are not obviously corrupted; the baseline is brittle under closed-loop visual/geometric distribution shift and lacks a robust final approach/grasp correction.

## Environment notes

- The first gate 0.065 server attempt inside the managed sandbox could not see CUDA/Vulkan (`Driver Version: 0`, `No device could be created`). Running the Isaac server in the real host environment saw RTX 3060/Vulkan correctly.
- The LeRobot client also needed real host execution to keep `policy.device=cuda`.
- Client was run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to use local SmolVLM2 cache and avoid network HEAD retries.
- Isaac server still aborts during cleanup with the known `tiled_camera.py` / replicator weak-reference error after artifacts are saved.

## Suggested next test

Run one diagnostic with an even more permissive close trigger or scripted final-centering aid, but keep it clearly separated from policy evaluation:

```bash
PORT=5571 CONTROL_MODE=joint \
FIXED_CUBE_POSE=0.425,-0.115,0.022 \
GRIPPER_CLOSE_GATE_EEF_CUBE_DIST=0.07 \
TRAJECTORY_LOG=_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_fixedcube_gate007_seed1002_20260707/baseline_eef_trajectory.jsonl \
_runtime/remote_handoff_gripper_lstm_work/experiment/run_eval_server.sh
```

Then run the same single-episode `lerobot-eval` command against port `5571` with offline HF env vars and `policy.path=_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`.
