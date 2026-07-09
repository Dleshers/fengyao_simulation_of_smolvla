# 2026-07-09 gated/zero-init torque gate1 evaluation

## Objective

Evaluate whether adding the frozen gripper-torque LSTM feature through a gated, zero-initialized adapter destabilizes the already-working visual SmolVLA pick-and-place policy.

The current purpose of this pick-and-place task is a non-collapse / integration check. Because the pure visual policy is already strong and the task is not very tactile-dependent, improvement is not expected to be a reliable signal here.

## Model under test

- Base checkpoint: `targeted5_visual_50k_seed1000_20260708/checkpoints/050000/pretrained_model`
- Gated torque continuation checkpoint:
  - `_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_seed1000_20260709/checkpoints/005000/pretrained_model`
- Torque encoder:
  - `trained_lstm_weights/torque_16d_encoder.pt`
- Torque LSTM configuration:
  - input size: `1`
  - hidden size: `32`
  - layers: `1`
  - output size: `16`
  - frozen: yes
- Injection configuration:
  - `torque_zero_init_adapter=True`
  - `torque_gate_init=1.0`
  - learned checkpoint gate: approximately `0.988`
  - adapter weights became non-zero after 5k continuation, so this was not the previous gate0 dead branch.

## Evaluation command

Server:

```bash
PORT=5562 bash experiment/eval_gated_torque_n10.sh server
```

Client:

```bash
PORT=5562 bash experiment/eval_gated_torque_n10.sh client
```

Evaluation output:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_eval_n10_seed1000_20260709
```

## Result

Gate1 gated torque result:

```text
successes = [True, True, False, True, True, True, False, True, True, True]
success rate = 8 / 10 = 80%
```

This matches the targeted5 visual 50k baseline and the visual-only 5k continuation control:

| model | success sequence | success rate |
|---|---:|---:|
| targeted5 visual 50k | `[T,T,F,T,T,T,F,T,T,T]` | 8/10 |
| targeted5 visual 50k + visual-only 5k | `[T,T,F,T,T,T,F,T,T,T]` | 8/10 |
| targeted5 visual 50k + ungated torque-LSTM 5k | `[T,T,F,T,T,T,F,F,T,T]` | 7/10 |
| targeted5 visual 50k + gated/zero-init torque-LSTM gate1 5k | `[T,T,F,T,T,T,F,T,T,T]` | 8/10 |

## Success consistency audit

Audit command:

```bash
python3 experiment/audit_eval_success_consistency.py \
  --eval-info _runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_eval_n10_seed1000_20260709/eval_info.json \
  --trajectory _runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_eval_n10_seed1000_20260709/trajectory.jsonl
```

Output:

```text
episode,eval_info_success,jsonl_is_success,termination_success,hit_max_steps,env_step_done,disagree
0,True,True,True,False,True,False
1,True,True,True,False,True,False
2,False,False,True,True,False,True
3,True,True,True,False,True,False
4,True,True,True,False,True,False
5,True,True,True,False,True,False
6,False,False,True,True,False,True
7,True,True,True,False,True,False
8,True,True,True,False,True,False
9,True,True,True,False,True,False
```

Episodes 2 and 6 are timeout failures. As in previous audits, `termination_terms.success=True` can appear on timeout failures and should not be used as the authoritative success criterion. Use `eval_info.json` and the trajectory top-level `is_success` instead.

## Notes

- The Isaac eval server still aborts during shutdown with the known tiled-camera / syntheticdata weakref cleanup error after results have already been written.
- This cleanup abort does not invalidate this run because `eval_info.json`, `trajectory.jsonl`, and videos were already produced.
- Videos are under:

```text
_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/targeted5_visual50k_plus_gated_torque_lstm_gate1_5k_eval_n10_seed1000_20260709/videos/isaaclab_tactile_remote_0/
```

## Interpretation

The gate1 gated/zero-init torque injection did not collapse the task and recovered the extra failure introduced by the earlier ungated torque run. On this pick-and-place benchmark, it also did not exceed the visual-only success rate.

Current conclusion:

1. Gated/zero-init torque injection is the safer integration method than direct ungated injection.
2. Pick-and-place is suitable as a feasibility and non-regression benchmark.
3. It is not a strong task for measuring tactile benefit because visual information is already sufficient for high success.
4. The next meaningful comparison should use a more tactile-sensitive task, while retaining this gate1 configuration as the default safe injection path.

