# Targeted5 visual-only 50k upload and closed-loop evaluation

Date: 2026-07-08 local / 2026-07-09 UTC

## Summary

The targeted5 visual-only SmolVLA retraining reached the intended 50k step
checkpoint, was uploaded to a private Hugging Face model repository, and was
evaluated in IsaacLab remote closed loop for 10 episodes with seed 1000.

Result: **8/10 success, 80% success rate**.

This is a large improvement over the original pure visual baseline formal
evaluation, which had 0/10 success in the prior dynamic RGB run, but it is
slightly lower than the earlier targeted5 5k smoke evaluation, which reported
9/10 success on the same 10-episode seed range.

## Model checkpoint

Local checkpoint:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model
```

Validation:

- `training_state/training_step.json`: `{"step": 50000}`
- `model.safetensors`: readable in the LeRobot environment
- safetensors tensor count: 500
- uploaded files: `config.json`, `model.safetensors`, pre/post processor JSON and safetensors, `train_config.json`

Hugging Face repository:

```text
Dleshers/smolvla-franka-pickplace-targeted5-50k-seed1000
```

Repository state after upload:

- private: true
- commit: `144c89417062186356b1053039346536396b625a`
- used storage: 906,721,236 bytes

The upload used the current `hf` CLI, not the deprecated `huggingface-cli`.

## Evaluation setup

Policy:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model
```

Output:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual_50k_eval_n10_seed1000_20260708
```

Key settings:

- `--env.type=isaaclab_tactile_remote`
- `--env.server_port=5562`
- `--env.control_mode=joint`
- `--env.observation_height=224`
- `--env.observation_width=224`
- `--env.include_gripper_torque_window=false`
- `--eval.n_episodes=10`
- `--eval.batch_size=1`
- `--seed=1000`
- rename map:

```json
{
  "observation.images.rgb_table": "observation.images.camera1",
  "observation.images.rgb_wrist": "observation.images.camera2"
}
```

IsaacLab server wrote trajectory telemetry to:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual_50k_eval_n10_seed1000_20260708/trajectory.jsonl
```

## Evaluation artifacts

Generated files:

- `eval_info.json`
- `trajectory.jsonl`
- 10 videos:
  - `videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_1.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_2.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_3.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_4.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_5.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_6.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_7.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_8.mp4`
  - `videos/isaaclab_tactile_remote_0/eval_episode_9.mp4`

Client log:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/logs/targeted5_visual_50k_eval_client_n10_seed1000_20260708.log
```

Server log:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/logs/targeted5_visual_50k_eval_server_n10_seed1000_20260708.log
```

## Metrics

`eval_info.json` successes:

```text
[true, true, false, true, true, true, false, true, true, true]
```

Overall:

- success: 8/10
- success rate: 80%
- eval time: 331.38 s
- average episode time: 33.14 s

Per episode telemetry summary:

| Episode | Seed | Success | Final step | Min EEF-cube distance | Final EEF-cube distance |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 1000 | true | 199 | 0.0368 | 0.2383 |
| 1 | 1001 | true | 196 | 0.0358 | 0.2736 |
| 2 | 1002 | false | 300 | 0.0510 | 0.1075 |
| 3 | 1003 | true | 205 | 0.0361 | 0.2651 |
| 4 | 1004 | true | 200 | 0.0401 | 0.2581 |
| 5 | 1005 | true | 208 | 0.0383 | 0.2730 |
| 6 | 1006 | false | 300 | 0.0533 | 0.1006 |
| 7 | 1007 | true | 196 | 0.0443 | 0.2683 |
| 8 | 1008 | true | 199 | 0.0433 | 0.2848 |
| 9 | 1009 | true | 201 | 0.0393 | 0.2710 |

The two failed episodes are `2` and `6`, corresponding to seeds `1002` and
`1006`.  Both reached the 300-step limit and had noticeably larger minimum
EEF-cube distances than most successful episodes.

Gripper command ranges remained in the expected approximately normalized
range, roughly `[-1.05, 1.04]`; no obvious gripper action-scale explosion was
observed from telemetry.

## Notes and caveats

1. IsaacLab server startup was slow because asset loading included remote
   Omniverse/material warnings. It eventually reached:

   ```text
   Server listening on tcp://*:5562
   ```

2. After the client completed and wrote all results, shutting down the Isaac
   server still produced the known tiled-camera cleanup abort:

   ```text
   ReferenceError: weakly-referenced object no longer exists
   ```

   This occurred after `End of eval` and after all videos, `eval_info.json`,
   and `trajectory.jsonl` had been written.

3. In the final telemetry row of the two timeout failures, `termination_terms`
   included `"success": true` while the exported top-level `is_success` and
   `eval_info.json` success were false. For this evaluation report, the
   authoritative result is `eval_info.json`: episodes 2 and 6 are failures.
   The termination-term mismatch should be audited before using trajectory
   terminal rows alone as the success source.

## Recommended next steps

1. Inspect videos for failed episodes:

   ```bash
   mpv _runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual_50k_eval_n10_seed1000_20260708/videos/isaaclab_tactile_remote_0/eval_episode_2.mp4
   mpv _runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual_50k_eval_n10_seed1000_20260708/videos/isaaclab_tactile_remote_0/eval_episode_6.mp4
   ```

2. Run a fixed-cube or targeted replay around seeds 1002 and 1006 to decide
   whether the remaining failures are coverage gaps, timeout/success-threshold
   issues, or post-grasp placement instability.

3. Audit the terminal-row `termination_terms.success` versus `is_success`
   mismatch in `eval_server.py` before relying on trajectory JSONL alone for
   automated success labels.

