# 2026-07-09 visual-control vs torque-LSTM final audit

## Purpose

验证 pure visual targeted5 50k 已经成功后，继续训练 5k 时：

1. 仅做 visual-only continuation 是否会改变闭环表现；
2. 在同样 base checkpoint、同样 dataset、同样 seed/batch/steps 下，加入 frozen gripper torque-LSTM 注入 action expert 是否带来收益或扰动。

## Compared runs

| run | checkpoint | eval directory |
|---|---|---|
| visual50k | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model` | `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual_50k_eval_n10_seed1000_20260708` |
| visual50k + visual-only 5k control | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual50k_plus_visualonly_5k_seed1000_20260708/checkpoints/005000/pretrained_model` | `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual50k_plus_visualonly_5k_eval_n10_seed1000_20260709` |
| visual50k + torque-LSTM 5k | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual50k_plus_torque_lstm_5k_seed1000_20260708/checkpoints/005000/pretrained_model` | `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual50k_plus_torque_lstm_5k_eval_n10_seed1000_20260708` |

The visual-only control checkpoint was verified to contain 500 tensors and no
`torque*` tensors.

## Eval protocol

- env: `isaaclab_tactile_remote`
- control mode: joint
- episodes: 10
- seed range: 1000-1009
- cameras: table + wrist RGB, renamed to `observation.images.camera1` and `observation.images.camera2`
- visual-only eval does not send gripper torque window
- torque eval sends gripper torque window, size 30

Isaac cleanup still aborts after eval shutdown with the known weakref/tiled-camera
cleanup issue. This happened after `eval_info.json`, videos, and trajectory logs
were already written.

## Results

| run | successes | failed episodes | success rate |
|---|---|---|---:|
| visual50k | `TTFTTTFTTT` | 2, 6 | 8/10 = 80% |
| visual50k + visual-only 5k | `TTFTTTFTTT` | 2, 6 | 8/10 = 80% |
| visual50k + torque-LSTM 5k | `TTFTTTFFTT` | 2, 6, 7 | 7/10 = 70% |

The visual-only 5k continuation exactly preserves the visual50k success/failure
pattern. Therefore the torque-LSTM drop is not explained by "additional 5k
training" alone.

## Episode-level interpretation

Episodes 2 and 6 fail in all three runs. These are still hard visual/control
poses that targeted5 did not fully solve.

Episode 7 is the decisive ablation:

- visual50k succeeds
- visual50k + visual-only 5k succeeds
- visual50k + torque-LSTM 5k fails
- torque final eef-cube distance: about `0.0368 m`
- visual-only final eef-cube distance: about `0.2683 m` after successful reset/finish

This suggests the torque model does approach the cube, but gets stuck near the
object or perturbs the grasp/place sequence enough to timeout.

## Action-difference observations

Generated analysis files:

- `experiment_results/2026-07-09_torque_lstm_vs_visual_control_audit/summary.csv`
- `experiment_results/2026-07-09_torque_lstm_vs_visual_control_audit/episode_outcomes.csv`
- `experiment_results/2026-07-09_torque_lstm_vs_visual_control_audit/visualonly5k_vs_torque5k_action_diff.csv`
- `experiment_results/2026-07-09_torque_lstm_vs_visual_control_audit/episode_action_key_stats.csv`

For episode 7, comparing torque-LSTM 5k against the matched visual-only 5k
control over paired steps:

- action dim 6 MAE: about `0.0996`
- gripper dim 7 MAE: about `0.0838`
- action dim 1 max difference: about `0.6803`
- gripper dim 7 max difference: about `1.9610`

The effect is not a global numeric explosion: several successful episodes also
show transient gripper differences. But episode 7 has a combination of elevated
joint/action dim 6 and gripper perturbation, and is the only episode whose
outcome flips from success to failure.

## Success bookkeeping audit

The previous success telemetry caveat remains important:

- `eval_info.json` and top-level JSONL `is_success` agree.
- On timeout failures, the final JSONL row can still show
  `termination_terms.success=true`.
- Those timeout rows also show `hit_max_steps=true` and `env_step_done=false`.

Use `eval_info.json` / top-level `is_success` as authoritative. The stale or
non-final `termination_terms.success` field should not be used as the final
success criterion.

## Final conclusion

Current evidence favors:

1. Pure visual targeted5 50k is the strongest current model among these runs.
2. Continuing visual-only for 5k does not hurt and does not improve: it remains
   at 80%.
3. Frozen torque-LSTM injection into the action expert is technically functional
   but currently hurts closed-loop performance: 70% vs the matched visual-only
   control's 80%.
4. The degradation is localized rather than catastrophic. The torque branch does
   not break model loading, inference, action scale, checkpoint saving, or basic
   movement. It appears to introduce a mild but behaviorally meaningful action
   perturbation around grasp/place timing.

## Recommendation

Do not launch a full 50k torque-LSTM retrain with the current injection strength
as the next default experiment.

Recommended next experiment:

- keep the frozen torque encoder;
- initialize `torque_to_expert` near zero, or add a scalar gate initialized to
  zero/small value;
- run a short 5k ablation first;
- require success to be at least equal to the visual-only 5k control before
  spending a full 50k run.

Concrete next command should be a short gated-torque training run, not another
ungated 50k torque run.
