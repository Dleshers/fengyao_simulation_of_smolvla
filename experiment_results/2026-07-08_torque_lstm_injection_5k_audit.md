# 2026-07-08 torque-LSTM injection 5k audit

## Question

验证在已成功的 targeted5 pure visual 50k SmolVLA 基线之上，加入夹爪力矩窗口，经 frozen LSTM 编码后注入 action expert，是否会改善闭环表现，还是扰乱模型行为。

## Models compared

- Visual baseline:
  - local checkpoint: `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model`
  - previous eval: `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual_50k_eval_n10_seed1000_20260708`
- Torque-LSTM 5k continuation:
  - base checkpoint: same visual 50k checkpoint above
  - local checkpoint: `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual50k_plus_torque_lstm_5k_seed1000_20260708/checkpoints/005000/pretrained_model`
  - eval: `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual50k_plus_torque_lstm_5k_eval_n10_seed1000_20260708`

## Torque configuration

Authority configuration was used:

- `torque_input_dim=1`
- `torque_lstm_hidden_dim=32`
- `torque_lstm_num_layers=1`
- `torque_lstm_output_dim=16`
- `train_torque_lstm=false`
- `use_torque_lstm=true`
- `use_tactile=false`
- weights: `trained_lstm_weights/torque_16d_encoder.pt`

Checkpoint validation passed:

- `torque_lstm.*`: 6 tensors
- `torque_norm.*`: 2 tensors
- `torque_to_expert.*`: 2 tensors
- full checkpoint: 510 tensors
- `model.torque_lstm.lstm.weight_ih_l0`: `(128, 1)`
- `model.torque_lstm.lstm.weight_hh_l0`: `(128, 32)`
- `model.torque_lstm.fc.weight`: `(16, 32)`

The known `safetensors` shared-storage issue still appears at save time for `model.torque_lstm.lstm.weight_ih_l0`, but the patched fallback saves a cloned contiguous state dict and training completes.

## Evaluation result

Same eval seed range was used (`seed=1000`, 10 episodes).

| model | successes | success rate |
|---|---:|---:|
| visual targeted5 50k | `[T,T,F,T,T,T,F,T,T,T]` | 8/10 = 80% |
| visual50k + torque-LSTM 5k | `[T,T,F,T,T,T,F,F,T,T]` | 7/10 = 70% |

The torque-LSTM run keeps the two existing visual failures:

- episode 2 / seed 1002
- episode 6 / seed 1006

It also introduces one new failure:

- episode 7 / seed 1007, which visual baseline succeeded.

## Action and trajectory observations

Generated analysis files:

- `experiment_results/2026-07-08_torque_lstm_injection_audit/episode_summary.csv`
- `experiment_results/2026-07-08_torque_lstm_injection_audit/action_dim_stats.csv`
- `experiment_results/2026-07-08_torque_lstm_injection_audit/paired_action_diff_by_episode.csv`

Key observations:

- No obvious action explosion was observed. Gripper command stayed around `[-1.05, 1.07]`.
- Failed episodes still approach the cube, but do not complete a successful pick/place within 300 steps.
- The newly failing episode 7 has a notably larger paired action shift than most successful episodes:
  - joint/action dim 6 MAE about `0.100`
  - gripper dim 7 MAE about `0.082`
  - torque run final eef-cube distance about `0.0368`, i.e. it stays close to the cube rather than completing.
- For seed1002 and seed1006, both visual and torque versions fail; torque does not repair these hard poses.

Interpretation: after only 5k steps, torque injection is operational but not yet beneficial in closed loop. It appears to perturb the action expert enough to lose one previously successful pose, while not fixing the old failure poses. This is not a catastrophic integration failure, but it is a negative early ablation result relative to the pure visual targeted5 50k baseline.

## Known evaluation caveat

Trajectory telemetry still has a success bookkeeping inconsistency on timeout failures:

- top-level `is_success=false`
- `eval_info.json` success list reports failure
- but the final `termination_terms.success` can be `true`

For reported success rate, use `eval_info.json` / `is_success` as authoritative. This mismatch should be fixed before publishing final benchmark numbers.

Follow-up audit on 2026-07-08 confirmed the pattern for both visual 50k and
torque-LSTM 5k:

- successful episodes end with `env_step_done=true`, `is_success=true`, and
  `termination_terms.success=true`
- timeout failures end with `hit_max_steps=true`, `env_step_done=false`,
  `is_success=false`, while `termination_terms.success` can still be `true`

Therefore the 80% vs 70% comparison is not explained by a success-statistics
bug. The final LeRobot metric and top-level JSONL `is_success` agree; the stale
or non-final `termination_terms.success` field is only misleading for manual
trajectory inspection.

Small diagnostic patch applied:

- `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/scripts/eval_server.py`
  now writes `success_source` and
  `success_term_disagrees_with_reported` to both trajectory JSONL and env info.

Helper script added:

- `experiment/audit_eval_success_consistency.py`

This keeps `eval_info.json` / `is_success` as the authoritative success result,
while making termination-term disagreements explicit.

Isaac cleanup still aborts after results are written:

- `ReferenceError: weakly-referenced object no longer exists`
- location: tiled camera cleanup / synthetic data detach

This occurs after `eval_info.json`, videos, and `trajectory.jsonl` are saved.

## Current conclusion

The present evidence favors:

1. The torque-LSTM integration path works technically: load, train, save, instantiate, and eval all run.
2. Frozen torque-LSTM injection into the action expert does not immediately destabilize the model numerically.
3. At 5k continuation, it does not improve the visual baseline and likely causes mild action confusion on at least one pose.
4. Before launching a 50k torque-LSTM retrain, run a matched visual-only 5k continuation control from the same visual 50k checkpoint on the same targeted5 dataset. This isolates whether the 70% result is caused by torque injection or by any additional 5k fine-tuning on the expanded dataset.

## Recommended next command

Run a matched visual-only 5k continuation using the same dataset, seed, batch size, scheduler, and base checkpoint, but with `--policy.use_torque_lstm=false`, then evaluate with the same 10 seeds.

Prepared scripts:

- train: `experiment/train_visual50k_plus_visualonly_5k_control.sh`
- eval: `experiment/eval_visual5k_control_n10.sh`

The attempted launch was blocked by the current execution approval/usage limit,
so the matched-control training has not yet run in this session.
