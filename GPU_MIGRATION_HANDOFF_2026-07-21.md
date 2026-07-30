# GPU/storage migration handoff for the next Codex agent (2026-07-21)

> Audience: Codex on the new AutoDL/GPU machine after the storage disk from this machine has been attached or copied. Read this before running experiments.

## 1. Migration situation

The user is moving the storage from the current AutoDL instance to another GPU instance because the current instance cannot perform Isaac Sim headless RGB camera capture reliably. The current machine can run physics and non-visual/tactile/state experiments, but online RGB observations are blocked by the local Isaac Sim 4.5 RTX/Hydra/viewport stack.

Current failing machine evidence:

```text
GPU: NVIDIA GeForce RTX 4080 SUPER
Driver observed before migration: 595.71.05
/root/autodl-tmp capacity after expansion: 100G, ~55G free at last check
Isaac env: _runtime/remote_handoff_gripper_lstm_work/.conda/isaaclab
LeRobot env: _runtime/remote_handoff_gripper_lstm_work/.venv/lerobot
```

If choosing among AutoDL driver images, prefer a less-new driver image first. From the options discussed with the user, `560.35.03 / CUDA <= 12.6` is the best first candidate. Avoid another `595.x` image for this Isaac Sim 4.5 camera issue unless no alternative exists.

## 2. What has already been completed on the source machine

### Environment and dependency work

- IsaacLab/LeRobot environments were provisioned under `_runtime/remote_handoff_gripper_lstm_work`.
- Hugging Face login was available as user `Dleshers` with active token name `remote_server` during the previous run. Recheck with `hf auth whoami` after migration.
- Large downloads are intentionally kept outside normal Git under `_runtime/`, `persistent/`, and HF/local caches. Do not commit these.

### Training/evaluation already run

- Completed torque-disambiguation 2000-step comparison. See:
  - `experiment_results/2026-07-20_torque_disambiguation_2000step_eval.md`
- Completed peg-insert Stage 0/1 no-camera feasibility/data smoke. See:
  - `experiment_results/2026-07-20_peg_insert_stage0_stage1_nocam_feasibility.md`
- Recorded current headless camera diagnosis. See:
  - `experiment_results/2026-07-20_headless_camera_capture_status.md`

### Important model artifact

The user supplied the standalone torque LSTM weight path:

```text
/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla/trained_lstm_weights/torque_16d_encoder.pt
```

After storage migration, verify this file still exists before torque training/evaluation.

## 3. Source-tree changes included in this handoff

The following Git-tracked files were updated or added for the migration/debug state:

```text
experiment/peg_insert_env_smoke.sh
experiment/peg_insert_headless_probe.py
experiment/record_peg_insert_oracle_demos.py
experiment/train_peg_insert_torque_mvp_smoke.sh
experiment/replicator_min_camera_probe.py
experiment_results/2026-07-20_headless_camera_capture_status.md
experiment_results/2026-07-20_peg_insert_stage0_stage1_nocam_feasibility.md
experiment_results/2026-07-20_torque_disambiguation_2000step_eval.md
patches/2026-07-21_autodl_headless_camera_runtime_diagnostics.patch
```

Key behavior changes:

- `peg_insert_env_smoke.sh` now defaults to no-camera headless physics smoke (`ENABLE_CAMERAS=0`) and chooses the non-rendering kit by default. Set `ENABLE_CAMERAS=1` explicitly for camera tests.
- `peg_insert_headless_probe.py` and `record_peg_insert_oracle_demos.py` remove camera sensors when cameras are disabled. When cameras are enabled, they force peg-insert cameras to RGB-only because the current experiment only consumes RGB and depth annotators failed first on the source machine.
- `train_peg_insert_torque_mvp_smoke.sh` removed unsupported policy flags `--policy.torque_zero_init_adapter` and `--policy.torque_gate_init`.
- `experiment/replicator_min_camera_probe.py` is a standalone cube + Replicator RGB camera smoke test. Run it before full peg_insert visual collection/evaluation.

## 4. Runtime patches present on the transferred storage

Some diagnostics touched files inside the runtime rather than Git source. They should be treated as debug patches, not final architecture. Their unified diff is saved in:

```text
patches/2026-07-21_autodl_headless_camera_runtime_diagnostics.patch
```

Most important runtime changes currently present if the disk is copied as-is:

1. `IsaacLab-Tactile/source/isaaclab/isaaclab/sim/simulation_context.py` has a guard for `get_active_viewport() is None`.
2. Replicator's `omni.replicator.core` extension has `omni.kit.viewport.window` commented out to avoid the source-machine segfault path. This prevents crashing but also caused empty RGB buffers on the old machine.
3. `omni.kit.widget.viewport/.../texture.py` has an environment-variable diagnostic skip: `PEG_INSERT_SKIP_VIEWPORT_WIDGET_HYDRA=1`. This avoided immediate segfault in one path but then hung; do not rely on it as a solution.

Backups exist beside the changed files with names like `.bak_20260720_*`. On a new GPU/driver image, first test both the as-transferred state and the restored official state if necessary. If official rendering works after driver migration, prefer restoring the Replicator/viewport debug edits and using the official `isaaclab.python.headless.rendering.kit`.

## 5. First checks after storage arrives on the new GPU machine

Run these before any long experiment:

```bash
cd /root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla
ROOT=/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla/_runtime/remote_handoff_gripper_lstm_work

df -h /root/autodl-tmp /tmp
nvidia-smi
hf auth whoami || true
ls -lh trained_lstm_weights/torque_16d_encoder.pt
```

Verify no stale Isaac/Python jobs survived the migration:

```bash
ps -eo pid,ppid,stat,cmd | grep -E 'isaac|peg_insert|replicator_min|lerobot' | grep -v grep || true
```

## 6. Camera validation sequence

### A. Standalone minimal RGB camera probe

This is the fastest pass/fail test for true visual capability. It creates a cube and a Replicator RGB render product. Passing means non-empty `84x84x3` or `84x84x4` arrays are printed. Empty `(0,)` means camera capture is still unusable.

```bash
cd /root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla
ROOT=/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla/_runtime/remote_handoff_gripper_lstm_work
mkdir -p "$ROOT/persistent/logs" /tmp/svl
export TMPDIR=/tmp/svl TMP=/tmp/svl TEMP=/tmp/svl OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
export CONDA_PREFIX="$ROOT/.conda/isaaclab"
export PATH="$ROOT/.conda/isaaclab/bin:$PATH"

timeout 240 "$ROOT/.conda/isaaclab/bin/python" \
  experiment/replicator_min_camera_probe.py \
  --device cuda:0 --headless --enable_cameras \
  --experience=isaacsim_4_5/isaaclab.python.headless.rendering.kit \
  2>&1 | tee "$ROOT/persistent/logs/replicator_min_camera_new_gpu.log"
```

If this crashes like the old machine, try the custom minimal kit only as a diagnostic:

```bash
timeout 240 "$ROOT/.conda/isaaclab/bin/python" \
  experiment/replicator_min_camera_probe.py \
  --device cuda:0 --headless --enable_cameras \
  --experience=isaacsim_4_5/isaaclab.python.headless.camera_minimal.kit \
  2>&1 | tee "$ROOT/persistent/logs/replicator_min_camera_custom_new_gpu.log"
```

Do not proceed to visual peg-insert evaluation until this standalone probe returns non-empty images.

### B. Peg-insert camera smoke

Only after A passes:

```bash
cd /root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla
ROOT=/root/autodl-tmp/simulation_smolvla/fengyao_simulation_of_smolvla/_runtime/remote_handoff_gripper_lstm_work \
ENABLE_CAMERAS=1 NUM_STEPS=4 TIMEOUT_SECONDS=300 \
EXPERIENCE=isaacsim_4_5/isaaclab.python.headless.rendering.kit \
bash experiment/peg_insert_env_smoke.sh

```

Acceptance: `reset_obs.rgb_camera.wrist_cam` and step image stats must be non-empty image tensors, not `(1, 0)`.

## 7. Next work after camera validation

1. If visual camera works, collect a small real RGB peg-insert oracle dataset first (2-3 demos), inspect HDF5 image datasets, then convert to LeRobot.
2. Train or evaluate insert-focused policies using only verified datasets.
3. For final comparison, use matched datasets/seeds and record visual-only vs torque-LSTM results.
4. Save every run log under `$ROOT/persistent/logs` and summarize results in `experiment_results/`.
5. Push only code/docs/small result markdown to GitHub; do not push `_runtime`, conda envs, HF caches, HDF5 datasets, model checkpoints, or logs unless intentionally curated.

## 8. If camera still fails on the new machine

Then continue non-visual/tactile/state experiments on this storage and report that online RGB Isaac camera validation remains blocked by the GPU image/driver stack. The no-camera path is known to work and is still useful for torque/state feasibility, but it cannot substitute for final visual closed-loop validation.

## 9. New GPU validation update (2026-07-21)

After migration to RTX 4090 with driver `560.35.03`, headless RGB capture is working with `isaaclab.python.headless.rendering.nongx.kit`, which disables NGX/DLSS. See `experiment_results/2026-07-21_new_gpu_headless_camera_validation.md`. Both standalone Replicator RGB and peg_insert `wrist_cam`/`table_cam` smoke tests passed, followed by a 1-demo RGB HDF5 collection and LeRobot conversion/audit.
