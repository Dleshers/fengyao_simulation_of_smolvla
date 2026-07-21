# 2026-07-20 torque-disambiguation 2000-step evaluation

## Scope

This evaluates the matched 2000-step offline diagnostic runs requested for the insert/peg torque-disambiguation dataset.  The dataset is a causal diagnostic dataset: identical visual/state streams can require different actions depending on gripper torque.  These numbers are therefore training-curve/action-label disambiguation evidence, not physical peg-insertion success rates.

## Runs

| Arm | Dataset/control | Checkpoint |
| --- | --- | --- |
| visual | original dataset, torque disabled | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_torque_disamb_visual_steps2000_seed1000_20260720r5/checkpoints/002000/pretrained_model` |
| torque | original aligned torque, torque LSTM enabled | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_torque_disamb_torque_steps2000_seed1000_20260720r11/checkpoints/002000/pretrained_model` |
| zero | zero torque control, torque LSTM enabled | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_torque_disamb_zero_steps2000_seed1000_20260720r11/checkpoints/002000/pretrained_model` |
| shuffle | globally shuffled torque control, torque LSTM enabled | `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/peg_insert_torque_disamb_shuffle_steps2000_seed1000_20260720r11/checkpoints/002000/pretrained_model` |

All four runs produced `checkpoints/002000/pretrained_model/model.safetensors`, `config.json`, and `train_config.json`.

## Loss Summary

| Arm | Logged points | First loss | Final loss | Min loss | Last-10 mean | Last-20 mean | Final grad norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual | 200 | 0.292 | 0.027 | 0.018 | 0.0271 | 0.02885 | 0.633 |
| torque | 200 | 0.454 | 0.001 | 0.001 | 0.0024 | 0.00360 | 0.147 |
| zero | 200 | 0.455 | 0.023 | 0.019 | 0.0251 | 0.02590 | 0.572 |
| shuffle | 200 | 0.455 | 0.023 | 0.019 | 0.0251 | 0.02595 | 0.571 |

## Interpretation

Correctly aligned torque has a large effect in the intended diagnostic setting.  Using last-10 mean loss, the real-torque arm is:

- `0.0024 / 0.0271 = 8.9%` of visual-only loss, a 91.1% reduction.
- `0.0024 / 0.0251 = 9.6%` of zero-torque loss, a 90.4% reduction.
- `0.0024 / 0.0251 = 9.6%` of shuffled-torque loss, a 90.4% reduction.

The zero and shuffled controls are essentially identical, which supports the intended conclusion: the improvement is not just from adding the torque branch or extra parameters; it depends on aligned torque information.

## Caveats

These are single-seed offline training diagnostics on a constructed causal dataset.  They validate that the torque-LSTM path can use torque to disambiguate labels, but they do not prove physical insert success.  A later physical/Isaac rollout benchmark should use a dataset or environment whose success predicate reflects actual insertion, slip, jam, regrasp, and retreat outcomes.
