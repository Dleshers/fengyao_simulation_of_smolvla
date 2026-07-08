# 2026-07-08 eval success plumbing fix

## Scope

Focused fix for baseline evaluation success accounting.  No training, dataset
overwrite, checkpoint overwrite, upload, or broad batch evaluation was started.

The goal was to prevent two opposite errors:

1. timeout episodes being counted as success because of stale termination terms;
2. genuine server-reported success being missed by LeRobot because it was not
   exposed through the exact `final_info` path expected by the evaluator.

## Files changed in the runtime workspace

- `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/scripts/eval_server.py`
- `_runtime/remote_handoff_gripper_lstm_work/lerobot-tactile/src/lerobot/envs/isaaclab_tactile_remote.py`
- `_runtime/remote_handoff_gripper_lstm_work/lerobot-tactile/src/lerobot/scripts/lerobot_eval.py`

Note: these files live under `_runtime/`; ordinary repository `git diff` may not
show them if that tree is ignored.  The active LeRobot import path was verified
as:

`_runtime/remote_handoff_gripper_lstm_work/lerobot-tactile/src/lerobot/...`

## Success semantics after fix

The evaluation path now distinguishes:

- `is_success`: physical task success only;
- `terminated`: true Isaac environment terminal transition, such as task
  success or failure/drop termination;
- `truncated`: max-step timeout;
- `hit_max_steps`: explicit server-side timeout marker;
- `env_step_done`: whether Isaac itself returned terminal/truncated on this
  step;
- `post_step_scene_was_auto_reset`: true only when Isaac auto-reset occurred
  after an actual environment terminal transition.

The LeRobot success reader now uses this order:

1. `info["final_info"]["is_success"]`;
2. `info["final_info"]["success"]`;
3. per-step `info["is_success"]`;
4. per-step `info["success"]`;
5. default false.

## Validation

Static checks:

```bash
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/python -m py_compile \
  _runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/scripts/eval_server.py \
  _runtime/remote_handoff_gripper_lstm_work/lerobot-tactile/src/lerobot/envs/isaaclab_tactile_remote.py \
  _runtime/remote_handoff_gripper_lstm_work/lerobot-tactile/src/lerobot/scripts/lerobot_eval.py
```

Result: passed.

Unit-level smoke checks:

- terminal success response becomes `terminated=True`, `truncated=False`,
  `final_info.is_success=True`;
- timeout failure response becomes `terminated=False`, `truncated=True`,
  `final_info.is_success=False`;
- `_extract_successes_from_info()` handles `final_info`, `is_success`, and
  `success` payloads.

Result: passed.

## Isaac/LeRobot 1-episode smoke eval

Output directory:

`_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/success_plumbing_smoke_fixedcube_seed1400_20260708/`

Configuration:

- baseline pure SmolVLA checkpoint;
- `n_episodes=1`;
- fixed cube pose: `(0.50, -0.05, 0.022)`;
- joint control;
- dynamic RGB;
- seed `1400`.

LeRobot result:

- `pc_success: 0.0`;
- `successes: [False]`;
- video:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/success_plumbing_smoke_fixedcube_seed1400_20260708/videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`.

Final JSONL step:

```json
{
  "step": 300,
  "done": true,
  "is_success": false,
  "termination_terms": {
    "cube_dropping": false,
    "basket_dropping": false,
    "success": false
  },
  "env_step_done": false,
  "hit_max_steps": true,
  "post_step_scene_was_auto_reset": false,
  "reward": 0.0
}
```

This confirms that timeout no longer reuses stale success and no longer reports
auto-reset pollution.

## Remaining caveat

The Isaac shutdown path still emits a known cleanup-time
`pybind11::error_already_set / weakly-referenced object no longer exists` abort
after the client finishes and the server is closed.  The telemetry and eval
artifacts were written before cleanup.  This is noisy but separate from success
accounting.

## Recommended next command

Rerun a small controlled pose where physical success is expected or near-miss
success is plausible, then verify that a genuine `is_success=True` is propagated
to `eval_info.json`.  For example, use the previously promising low-x pose:

```bash
PORT=5584 CONTROL_MODE=joint \
TRAJECTORY_LOG=$PWD/_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/success_plumbing_positive_probe_seed1401_20260708/trajectory.jsonl \
FIXED_CUBE_POSE='0.35,-0.18,0.022' \
_runtime/remote_handoff_gripper_lstm_work/experiment/run_eval_server.sh visual
```

Then run a matching 1-episode `lerobot-eval` client against port `5584`.

## Follow-up positive probe

Completed after the timeout smoke test.

Output directory:

`_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/success_plumbing_positive_probe_seed1401_20260708/`

Configuration:

- baseline pure SmolVLA checkpoint;
- `n_episodes=1`;
- fixed cube pose: `(0.35, -0.18, 0.022)`;
- joint control;
- dynamic RGB;
- seed `1401`.

Result:

- LeRobot `pc_success: 100.0`;
- per-task `successes: [true]`;
- video:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/success_plumbing_positive_probe_seed1401_20260708/videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`.

Final JSONL step:

```json
{
  "step": 163,
  "done": true,
  "is_success": true,
  "termination_terms": {
    "cube_dropping": false,
    "basket_dropping": false,
    "success": true
  },
  "env_step_done": true,
  "hit_max_steps": false,
  "post_step_scene_was_auto_reset": true,
  "reward": 0.0
}
```

Additional physical telemetry:

- minimum EEF-cube distance after step: `0.0476 m`;
- maximum cube z after step: `0.2050 m`;
- first success step: `163`.

Conclusion: success plumbing is now verified in both directions:

1. timeout failure remains `success=false`;
2. genuine Isaac terminal success propagates to LeRobot `eval_info.json`.

The remaining baseline problem is therefore policy/data coverage and closed-loop
robustness, not success accounting.
