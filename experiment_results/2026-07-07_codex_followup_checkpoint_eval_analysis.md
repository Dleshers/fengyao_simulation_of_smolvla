# 2026-07-07 Codex follow-up: baseline audit, checkpoint fix, eval-server preflight

## Scope

Follow-up after pulling `origin/main` at commit `682bc90bd05962e001495cdcaf9cc067c5ffbe24`.

Requested constraints observed:

- No formal 50k torque-LSTM training was started.
- No batch/closed-loop evaluation was started.
- No existing dataset/checkpoint was overwritten.
- Hugging Face token was not printed or copied.
- Large runtime assets remain under ignored `_runtime/`.

## Host and environment

- Host: `ubuntu2204-System-Product-Name`
- GPU: NVIDIA GeForce RTX 3060, driver `570.211.01`, 12288 MiB
- Disk: `/dev/sda3`, 916G total, 806G available after dataset/model download
- LeRobot env: `_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot`
  - LeRobot `0.4.3`
  - torch `2.7.1+cu126`
  - CUDA visible: yes
  - pyzmq `27.1.0`
- Isaac env: `_runtime/remote_handoff_gripper_lstm_work/.conda/isaaclab`
  - IsaacLab `0.46.3`
  - torch `2.7.0+cu128`
  - CUDA visible: yes
  - pyzmq `27.1.0`

## Assets restored

- Dataset downloaded to:
  `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1`
- Baseline model downloaded to:
  `_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- Hugging Face baseline repo verified available:
  `Dleshers/smolvla-franka-pickplace-baseline-50k-seed1000`
  at `a526d99c4d40aa8a8632d8c3e93831891468f6bf`

## Dataset and normalizer audit

`remote_workspace/experiment/validate_dataset.py` passed:

- 200 episodes
- 41,276 frames
- `observation.state`: `[9]`
- `action`: `[8]`
- `observation.images.camera1`: `[3,224,224]`
- `observation.images.camera2`: `[3,224,224]`
- `observation.gripper_torque`: float32 `[30,1]`
- torque window index `-1` is newest

Training distribution summary:

- action dim 8 gripper command is binary only: `-1.0` for 26,761 frames, `+1.0` for 14,515 frames
- gripper command mean/std: `-0.2967 / 0.9550`
- arm action appears to be near-absolute joint targets, not large deltas:
  mean absolute `action[:7] - state[:7]` is approximately
  `[0.0154, 0.0395, 0.0184, 0.0178, 0.0216, 0.0342, 0.0392]`
- newest torque range/mean/std:
  `[-57.5694, 19.1751] / -27.7595 / 20.5471`
- baseline `policy_preprocessor` and `policy_postprocessor` stats match dataset stats for `observation.state` and `action`.

## Baseline closed-loop evaluation audit status

The new result document reports two completed baseline evals:

- `eval_visual_seed1000`: 10 episodes, success rate 0%
- `eval_visual_n5_seed1000`: 10 episodes, success rate 0%

However, the 20 evaluation videos, `eval_info.json`, and trajectory JSONL logs were not present in this local checkout/runtime. Therefore video-level failure classification could not be performed here. Specifically, this machine could not determine from video whether the arm approached the cube, gripper closed, cube was pushed/lifted/dropped, or whether gripper action was visibly inverted.

Based on available training data/config only, the highest-priority checks remain:

1. eval action dimension must be joint-control 8D;
2. gripper command semantics must preserve binary `-1/+1` as used in the dataset;
3. eval client must use rename map:
   `rgb_table -> observation.images.camera1`,
   `rgb_wrist -> observation.images.camera2`;
4. baseline normalizer/postprocessor must be loaded from the uploaded checkpoint, not regenerated from an eval-time schema.

### Local baseline validity check

A local, non-evaluation baseline smoke was run after the real torque-LSTM weight update. This did not start Isaac rollout.

Setup:

- policy: `_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- dataset: `_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1`
- LeRobot factory path: `make_dataset`, `make_policy`, `make_pre_post_processors`
- `policy.load_vlm_weights=false` so the local checkpoint weights are used without downloading VLM weights
- samples: dataset indices `[0, 10, 1000, 10000]`

Results:

- baseline checkpoint loads successfully with dataset-refreshed features;
- policy inputs are:
  - `observation.state`
  - `observation.images.camera1`
  - `observation.images.camera2`
  - `observation.tactile.force_grid`
- policy output is `action: [8]`;
- forward losses on the sampled batches:
  - `0.007328`
  - `0.141847`
  - mean `0.074588`
- first-step predicted actions, after checkpoint postprocessor unnormalization, are close to dataset actions:
  - mean absolute action error across 4 sampled frames:
    `[0.02584, 0.03259, 0.03460, 0.01856, 0.05937, 0.02734, 0.06163, 0.01711]`
  - predicted gripper commands:
    `[1.03059, 1.01190, -0.99322, -1.01918]`
  - target gripper commands:
    `[1.0, 1.0, -1.0, -1.0]`

Interpretation:

- the uploaded baseline checkpoint is loadable and numerically active;
- the baseline postprocessor restores action scale correctly;
- on-distribution sampled observations produce joint-target actions close to demonstrations;
- gripper command sign and scale are not obviously inverted in local inference.

### Eval bridge code audit

The local eval bridge was inspected:

- `IsaacLabTactilePolicyObservationProcessorStep` in `control_mode=joint` constructs state as:
  `[arm_joint_pos(7), gripper_qpos(2)]`, matching the `[9]` training state;
- `IsaacLabTactilePolicyActionProcessorStep` returns action unchanged in `control_mode=joint`;
- eval policy postprocessor runs before env postprocessor, so the baseline `policy_postprocessor` should unnormalize action before sending it to Isaac;
- `eval_pair.sh` uses the required rename map:
  `rgb_table -> observation.images.camera1`,
  `rgb_wrist -> observation.images.camera2`.

This locally rules out the most obvious local causes: broken checkpoint, missing normalizer, wrong action dimension, or gripper sign flip in the LeRobot-side joint-mode bridge.

Remaining likely failure causes require closed-loop assets/telemetry:

1. Isaac env initial distribution or object/camera placement may differ from collection despite matching schema.
2. The visual policy may overfit offline demonstrations and fail after small closed-loop deviations.
3. Isaac server may be applying gripper command semantics differently from the demonstration collector.
4. Remote videos/trajectory logs are still required to classify whether failures are approach failure, no-close, push-away, lift/drop, or release failure.

### Saved local offline inference audit

At the user's request, a larger local offline inference audit was run and saved under:

- `experiment_results/2026-07-07_baseline_local_inference_audit/`

Files:

- `summary.json`: aggregate metrics, phase metrics, and worst samples
- `samples.csv`: 600 per-sample rows with episode/frame/phase, raw normalized action, unnormalized predicted action, target action, per-dim errors, gripper sign flags
- `README.md`: short explanation

Sampling:

- 200 episodes
- 3 frames per episode:
  - 20% progress
  - 50% progress
  - 80% progress
- total: 600 local inference samples

Aggregate results:

- mean batch forward loss: `0.022345`
- action MAE by dim:
  `[0.01063, 0.02219, 0.01365, 0.00994, 0.01642, 0.02066, 0.02684, 0.00904]`
- arm L2 error:
  - mean `0.05612`
  - p50 `0.04063`
  - p90 `0.10245`
  - p99 `0.29456`
- gripper sign accuracy: `1.0000`
- gripper binary accuracy: `1.0000`
- gripper MAE: `0.00904`

By phase:

| Phase | Samples | Arm L2 mean | Gripper sign acc | Gripper MAE |
| --- | ---: | ---: | ---: | ---: |
| 20% | 200 | 0.03964 | 1.0000 | 0.01150 |
| 50% | 200 | 0.08313 | 1.0000 | 0.00760 |
| 80% | 200 | 0.04559 | 1.0000 | 0.00801 |

Interpretation:

- baseline local inference is strongly consistent with the demonstration dataset;
- gripper command sign is not a local model/processor failure;
- the largest offline arm errors occur around mid-episode, which is plausibly the grasp/transport transition;
- the `pred_outside_train_range_count_by_dim` is nonzero only for gripper because predictions slightly overshoot `±1` (for example `1.05` or `-1.03`), while signs remain correct. This is unlikely to explain 0% success by itself unless the Isaac action manager treats values just outside `[-1,1]` pathologically.

This strengthens the conclusion that the 0% closed-loop baseline result is more likely caused by closed-loop distribution shift, Isaac/eval environment mismatch, or server-side actuation semantics than by a corrupted baseline checkpoint.

### Stepwise baseline failure-cause audit

A focused stepwise audit was added:

- `experiment_results/2026-07-07_baseline_failure_stepwise_audit.md`
- `experiment_results/2026-07-07_baseline_1ep_diagnostic_eval.md`
- `experiment_results/2026-07-07_baseline_eef_telemetry_diagnostic.md`

Most important new finding:

- the baseline checkpoint uses `chunk_size=50` and `n_action_steps=50`;
- `SmolVLAPolicy.select_action()` only calls the model when its action queue is empty;
- at 20 Hz, closed-loop eval therefore executes about 2.5 seconds of queued actions per visual observation.

This is not a serialization/config bug, but it is now the strongest local hypothesis for why an otherwise valid offline imitation model can fail closed-loop: small visual/action errors can accumulate for a long open-loop chunk before the policy re-observes the scene.

One diagnostic baseline episode was then run against the local server on port `5562`:

- episodes: `1`
- success: `false`
- output: `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_seed1000_20260707_0629/`
- video: `videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`
- trajectory: `baseline_probe_trajectory.jsonl`
- summary: `trajectory_summary.json`

Key trajectory finding:

- the gripper command turned negative at step `61`;
- gripper qpos was physically closed by step `66`;
- cube final displacement was only `0.00344 m`;
- max cube lift delta was only `0.01098 m`;
- cube final XY distance to basket was `0.26882 m`.

Interpretation: in this episode the policy did close the gripper, but the cube was not stably grasped or transported. The failure is therefore more consistent with approach/grasp pose error or closed-loop chunk drift than with a gripper sign/scale failure.

The eval server telemetry was then extended with end-effector pose and eef/cube distance:

- patch archived as `patches/2026-07-07_eval_server_eef_telemetry.patch`
- runtime file patched: `_runtime/remote_handoff_gripper_lstm_work/IsaacLab-Tactile/scripts/eval_server.py`

A second 1-episode diagnostic with eef telemetry was run on port `5563`:

- output: `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1000_20260707_0643/`
- success: `false`
- trajectory: `baseline_eef_trajectory.jsonl`
- summary: `eef_trajectory_summary.json`

Key eef findings:

- initial eef-cube distance: `0.25557 m`
- minimum eef-cube distance: `0.07436 m` at step `63`
- first negative gripper command: step `61`
- eef-cube distance at first close command: `0.07723 m`
- first physical gripper close: step `66`
- eef-cube distance at physical close: `0.07568 m`
- max cube lift delta: `0.01132 m`
- final cube displacement: `0.00396 m`

Interpretation: the gripper closes while the end-effector is still roughly `7.5 cm` from the cube center, and the cube is only nudged/slightly lifted. This narrows the baseline failure to approach/grasp-pose/timing error rather than gripper actuation or action-scale failure.

## Checkpoint save failure: reproduced and fixed

A 1-step torque-LSTM training smoke was run with a dummy shape-correct encoder only to reproduce serialization. This was not used as an experiment checkpoint.

Reproduction:

- command used `steps=1`, `save_freq=1`, `batch_size=1`
- torque config: `input=1, hidden=32, layers=1, output=16, frozen`
- checkpoint save failed with the same safetensors error:
  `found no suitable name to keep for saving amongst: {'model.torque_lstm.lstm.weight_ih_l0'}`

Cause:

- before training, full policy `save_pretrained()` succeeds;
- after a CUDA LSTM forward/backward, `safetensors.torch.save_model()` sees `model.torque_lstm.lstm.weight_ih_l0` as shared/view storage not covering entire storage;
- the thrown exception is `RuntimeError`, not `ValueError`.

Fix applied in runtime LeRobot:

- `PreTrainedPolicy._save_pretrained()` first tries normal `safetensors.save_model`;
- if a shared/storage `RuntimeError` or `ValueError` is raised, it saves a cloned contiguous CPU `state_dict` via `safetensors.save_file`;
- unrelated exceptions are re-raised.

Verification:

- repeated 1-step torque-LSTM training with forced checkpoint save succeeded;
- saved checkpoint:
  `_runtime/remote_handoff_gripper_lstm_work/tmp/torque_lstm_save_smoke_1step_fixed/checkpoints/000001/pretrained_model/model.safetensors`
- fallback warning was emitted exactly at checkpoint save and training ended normally.

Patch archived:

- `patches/2026-07-07_lerobot_safetensors_save_fix.patch`

## Additional compatibility fix

`SmolVLAConfig.tactile_token_mode` was changed from `Literal[...]` to `str` in the override/runtime copy because the current `draccus` decoder cannot parse the saved checkpoint JSON when that field is annotated as `typing.Literal`.

The runtime validation still checks allowed values in `__post_init__`.

## Eval server status

Confirmed and fixed:

- IsaacLab env now has `pyzmq==27.1.0`
- LeRobot eval client env now has `pyzmq==27.1.0`
- runtime `run_eval_server.sh` was synced to support `TRAJECTORY_LOG`
- runtime `scripts/eval_server.py` was patched to accept `--trajectory-log` and open a JSONL telemetry file

Attempted smoke:

- launched visual joint server on port `5561`
- trajectory file was created:
  `_runtime/remote_handoff_gripper_lstm_work/tmp/eval_server_probe_trajectory.jsonl`
- server reached Isaac/Vulkan initialization and opened the trajectory file
- it did not reach ZMQ listening/reset before manual stop; it stalled during Joint TacEx env setup/asset loading
- latest warning before stop referenced an unresolved Omniverse/simready material asset:
  `physics_stone.usda`

Result:

- pyzmq issue is resolved;
- `TRAJECTORY_LOG` plumbing is present;
- full eval-server communication is not yet proven on this local machine because Isaac env setup did not complete within the smoke window.

## 2026-07-07 update: real LSTM weights restored

After the initial report, `origin/main` was advanced to:

- `e5391140f34d1e58de26e37d2be5467f87dfe37b`

The new tracked file was restored:

- `trained_lstm_weights/torque_16d_encoder.pt`
- size: 34K
- sha256: `a232658aad57c3e2e34ea8123a06043db3e5d31dab12439ee9592d2925ad0ef5`

`remote_workspace/experiment/verify_frozen_torque_encoder.py` passed:

- strict architecture: `input=1 hidden=32 layers=1 output=16`
- encoder frozen
- LayerNorm and Action Expert projection receive gradients
- `[B,30,1] -> [B,16] -> [B,1,D_expert]`

## Real-encoder checkpoint save smoke

A 10-step smoke run was executed using the real `trained_lstm_weights/torque_16d_encoder.pt`:

- `steps=10`
- `batch_size=1`
- `save_freq=10`
- `train_torque_lstm=false`
- output:
  `_runtime/remote_handoff_gripper_lstm_work/tmp/torque_lstm_real_encoder_save_smoke`

Result:

- training reached step 10;
- checkpoint save again triggered the expected `safetensors.save_model` shared/view-storage warning on `model.torque_lstm.lstm.weight_ih_l0`;
- the fallback cloned contiguous `state_dict` save succeeded;
- `verify_torque_checkpoint.py` passed:
  - `torque_lstm.`: 6 tensors
  - `torque_norm.`: 2 tensors
  - `torque_to_expert.`: 2 tensors

Saved smoke checkpoint:

- `_runtime/remote_handoff_gripper_lstm_work/tmp/torque_lstm_real_encoder_save_smoke/checkpoints/000010/pretrained_model/model.safetensors`

## Why full torque-LSTM 50k was not started

The real encoder is now present and the short checkpoint-save smoke passes. Formal 50k torque-LSTM training was still not started because the user had not explicitly confirmed starting the full training run after the smoke.

Required before formal run:

- apply `patches/2026-07-07_lerobot_safetensors_save_fix.patch` to the training LeRobot checkout;
- use `trained_lstm_weights/torque_16d_encoder.pt`;
- launch the controlled 50k run only after explicit confirmation.

## Current code/doc changes to keep

- `.gitignore`: ignores `_runtime/`, `datasets/`, `pretrained/`, and `remote_workspace/.cache/`
- `remote_handoff_gripper_lstm/lerobot_overrides/configuration_smolvla.py`: `Literal` -> `str` compatibility
- `patches/2026-07-07_lerobot_safetensors_save_fix.patch`: archived checkpoint-save fix
- this report

## 2026-07-07 update: `n_action_steps=1` baseline diagnostic

To test whether the 50-step queued action chunk was the sole closed-loop failure cause, a third local 1-episode baseline diagnostic was run with:

- `--policy.n_action_steps=1`
- same checkpoint: `_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000`
- same seed: `1000`
- same joint-control visual eval server, with eef telemetry enabled
- output: `_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_n1_seed1000_20260707/`

Artifacts:

- trajectory: `baseline_eef_n1_trajectory.jsonl`
- summary: `eef_n1_trajectory_summary.json`
- video: `videos/isaaclab_tactile_remote_0/eval_episode_0.mp4`
- sampled frames: `frame_001.jpg` ... `frame_005.jpg`

Result:

- success: `false`
- reward sum: `0.0`
- episode length: `300` steps
- initial eef-cube distance: `0.25557 m`
- minimum eef-cube distance: `0.08044 m` at step `60`
- first negative gripper command: step `57`
- eef-cube distance at first negative command: `0.09070 m`
- first physical gripper close: step `60`
- eef-cube distance at physical close: `0.08044 m`
- cube final displacement: `0.00020 m`
- cube max displacement: `0.00397 m`
- cube max lift: `0.00258 m`
- joint target tracking error mean: `0.03711`

Interpretation:

- Reducing `n_action_steps` from `50` to `1` did not rescue the baseline.
- In this seed-1000 episode, the fully reactive policy still closed the gripper while the end-effector was about `8 cm` from the cube center.
- The cube was barely moved, so this run is even less grasp-like than the previous `n_action_steps=50` eef telemetry run.
- Therefore the 50-step queue can still be a risk/exacerbating factor, but it is not the sole root cause of the observed baseline 0% success.
- Current best diagnosis is a closed-loop approach/grasp-pose/timing failure under the local Isaac evaluation distribution, despite the checkpoint being valid and highly consistent with the offline dataset.

Operational note:

- Isaac server again aborted during camera/replicator cleanup after data had been written:
  `ReferenceError: weakly-referenced object no longer exists`.
- This remains a cleanup-path issue, not an episode-data validity issue.
- GPU state was checked afterwards outside the sandbox and was healthy:
  RTX 3060, driver `570.211.01`, 12288 MiB total, about 320 MiB used.

## Recommended next command

Recommended next diagnostic command, not a formal batch eval:

```bash
# Re-run a single episode with a different seed to test whether the 7-9 cm
# premature-close distance is seed-specific or systematic.
PORT=5565 CONTROL_MODE=joint \
TRAJECTORY_LOG=_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1001_20260707/baseline_eef_trajectory.jsonl \
bash _runtime/remote_handoff_gripper_lstm_work/experiment/run_eval_server.sh visual
```

Then, in another shell:

```bash
HF_HOME=_runtime/remote_handoff_gripper_lstm_work/.cache/huggingface \
XDG_CACHE_HOME=_runtime/remote_handoff_gripper_lstm_work/.cache \
CONTROL_MODE=joint \
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/lerobot-eval \
  --policy.path=_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000 \
  --policy.load_vlm_weights=false \
  --policy.n_action_steps=1 \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost --env.server_port=5565 \
  --env.task=pick_place --env.torque_window_size=30 \
  --env.observation_height=224 --env.observation_width=224 \
  --env.control_mode=joint \
  --rename_map='{"observation.images.rgb_table":"observation.images.camera1","observation.images.rgb_wrist":"observation.images.camera2"}' \
  --env.include_gripper_torque_window=false \
  --eval.n_episodes=1 --eval.batch_size=1 \
  --seed=1001 \
  --output_dir=_runtime/remote_handoff_gripper_lstm_work/persistent/eval_diagnostics/baseline_1ep_eef_seed1001_20260707
```

Formal torque-LSTM training should still only be started after explicit confirmation:

```bash
PYTHONPATH=_runtime/remote_handoff_gripper_lstm_work/lerobot-tactile/src \
TMPDIR=_runtime/remote_handoff_gripper_lstm_work/tmp \
_runtime/remote_handoff_gripper_lstm_work/.venv/lerobot/bin/lerobot-train \
  --dataset.repo_id=franka_pickplace_joint_visual_torque_w30_v1 \
  --dataset.root=_runtime/remote_handoff_gripper_lstm_work/persistent/datasets/franka_pickplace_joint_visual_torque_w30_v1 \
  --policy.path=_runtime/remote_handoff_gripper_lstm_work/pretrained/baseline_smolvla_50k_seed1000 \
  --policy.device=cuda \
  --policy.load_vlm_weights=false \
  --policy.push_to_hub=false \
  --policy.use_tactile=false \
  --policy.use_torque_lstm=true \
  --policy.torque_window_key=observation.gripper_torque \
  --policy.torque_window_size=30 \
  --policy.torque_input_dim=1 \
  --policy.torque_lstm_hidden_dim=32 \
  --policy.torque_lstm_output_dim=16 \
  --policy.torque_lstm_num_layers=1 \
  --policy.torque_lstm_weights_path=trained_lstm_weights/torque_16d_encoder.pt \
  --policy.train_torque_lstm=false \
  --seed=1000 --batch_size=1 --num_workers=0 \
  --steps=50000 --log_freq=100 --save_checkpoint=true --save_freq=5000 \
  --wandb.enable=false \
  --output_dir=_runtime/remote_handoff_gripper_lstm_work/persistent/gripper_lstm_experiments/torque_lstm_smolvla_50000_seed1000
```
