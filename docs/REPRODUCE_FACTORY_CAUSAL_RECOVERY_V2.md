# Reproduce and continue the Factory causal-recovery v2 experiment

This document is the authoritative procedure for the current experiment. It replaces all older top-level handoff/setup notes.

## What is reproduced

The experiment trains four matched 50k-step SmolVLA policies from the same base model and 120 successful Factory peg-in-hole demonstrations:

| Arm | Dataset torque | Policy torque token |
|---|---|---|
| `visual` | original dataset | disabled |
| `torque` | causal original torque | enabled, frozen LSTM |
| `zero` | all-zero torque | enabled, frozen LSTM |
| `shuffle` | episode-shuffled torque | enabled, frozen LSTM |

The causal v2 collector records `state_t`, RGB and torque before `action_t` is executed. Recovery episodes apply one unlabelled random lateral perturbation and then retain only Oracle corrective actions. This eliminates the action/next-observation mismatch in the earlier `formal_v1` data.

## Hardware, OS and storage

Validated host: Ubuntu 22.04, RTX 4090 24GB, NVIDIA driver `560.35.03`, CUDA-compatible PyTorch `2.7.0+cu128`, Python `3.10.8`. Headless RGB capture uses `isaaclab.python.headless.rendering.nongx.kit`.

Use at least 100GB writable local Linux storage; 120GB+ is recommended. The run needs about 2GB for three converted datasets and roughly 15GB for four checkpoints, in addition to Isaac Sim and model caches. Do not place conda environments, Isaac caches, or datasets on network/NTFS storage.

Do not downgrade to the previously tested `595.x` driver path: it did not provide reliable headless capture in this setup.

## Source versions and runtime layout

```bash
export REPO=$PWD/fengyao_simulation_of_smolvla
export ROOT=$REPO/_runtime/remote_handoff_gripper_lstm_work
mkdir -p "$ROOT"

git clone https://github.com/Dleshers/fengyao_simulation_of_smolvla.git "$REPO"
git clone https://github.com/rechim25/IsaacLab-Tactile.git "$ROOT/IsaacLab-Tactile"
git -C "$ROOT/IsaacLab-Tactile" checkout b33dbafb
git clone https://github.com/rechim25/lerobot-tactile.git "$ROOT/lerobot-tactile"
git -C "$ROOT/lerobot-tactile" checkout 8b81b44
```

Create/restore an Isaac-compatible Python 3.10 environment at `$ROOT/.conda/isaaclab`. The exact validated package set is [requirements/isaaclab_python_3.10.8_20260724.txt](../requirements/isaaclab_python_3.10.8_20260724.txt). Install Isaac Sim 4.5 according to NVIDIA's Linux pip installation, then install this package snapshot and the two source trees in editable mode. Keep the official SmolVLA base at:

```text
$ROOT/pretrained/official_smolvla_base
```

The standalone frozen torque encoder must be present at:

```text
$REPO/trained_lstm_weights/torque_16d_encoder.pt
```

Apply the runtime modifications after checking the source SHAs above:

```bash
git -C "$ROOT/IsaacLab-Tactile" apply "$REPO/patches/runtime_20260724/IsaacLab-Tactile-b33dbafb-runtime.patch"
git -C "$ROOT/lerobot-tactile" apply "$REPO/patches/runtime_20260724/lerobot-tactile-8b81b44-runtime.patch"
```

The patches include the torque-token implementation and the compatible headless/runtime edits. Do not apply them to another commit without resolving conflicts and re-running the smoke test.

## Required environment variables

```bash
export CONDA_PREFIX="$ROOT/.conda/isaaclab"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=yes
export ACCEPT_EULA=Y
export HF_HOME="$ROOT/.cache/huggingface"
export TMPDIR=/tmp/svl TMP=/tmp/svl TEMP=/tmp/svl
mkdir -p "$TMPDIR" "$ROOT/persistent/logs"
```

The EULA variables are only valid after accepting the NVIDIA Omniverse EULA. Use `setsid -f` or `nohup` for all long jobs so VSCode/SSH closure cannot terminate them.

## Preflight

```bash
cd "$REPO"
df -h /root/autodl-tmp /tmp
nvidia-smi
ls -lh trained_lstm_weights/torque_16d_encoder.pt
"$CONDA_PREFIX/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
```

Before collection, run the RGB camera smoke test with the non-NGX headless kit. It must produce non-empty images. Do not proceed on an empty image buffer.

```bash
"$ROOT/IsaacLab-Tactile/isaaclab.sh" -p experiment/factory_replicator_camera_smoke.py \
  --headless --enable_cameras \
  --experience "$ROOT/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit"
```

## Collection, conversion, audit and training

The collector and pipeline are resumable only at completed-demo boundaries. Never point two collectors at one HDF5 file.

```bash
RUN=factory_peg_insert_causal_recovery_v2_formal120_$(date +%Y%m%d)
RAW="$ROOT/persistent/raw_hdf5/$RUN/peg_insert_demos.hdf5"
mkdir -p "$(dirname "$RAW")"

setsid -f "$ROOT/IsaacLab-Tactile/isaaclab.sh" -p "$REPO/experiment/collect_factory_peg_insert_causal_recovery.py" \
  --headless --enable_cameras \
  --experience "$ROOT/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit" \
  --dataset-file "$RAW" --num-demos 120 --max-attempts 320 --recovery-fraction .5 --seed 20260724 \
  > "$ROOT/persistent/logs/${RUN}.log" 2>&1
```

For the validated run name `factory_peg_insert_causal_recovery_v2_formal120_20260724`, run the checked-in pipeline after collection. It validates 120 strict demos, requires at least 40 recovery episodes, produces all three local LeRobot datasets, audits them, then trains the four arms sequentially:

```bash
setsid -f "$REPO/experiment/run_factory_causal_recovery_v2_pipeline.sh" COLLECTOR_PID \
  > "$ROOT/persistent/logs/factory_causal_recovery_v2_pipeline.log" 2>&1
```

Replace `COLLECTOR_PID` with the `isaaclab.sh` launcher PID. The pipeline refuses incomplete raw data, invalid datasets and pre-existing output directories. Its training order is `visual → torque → zero → shuffle`.

## Evaluation

Use only completed `checkpoints/050000/pretrained_model` directories. All arms must use the same seeds, strict predicate, camera setup and `n_action_steps`. The following is the canonical one-arm form:

```bash
POLICY="$ROOT/persistent/gripper_lstm_experiments/factory_causal_recovery_v2_120demos_20260724_r1_visual_50k_seed1000/checkpoints/050000/pretrained_model"
DATA="$ROOT/persistent/lerobot_datasets/Dleshers/factory-peg-insert-causal-recovery-v2-original"
"$ROOT/IsaacLab-Tactile/isaaclab.sh" -p "$REPO/experiment/eval_factory_peg_insert_formal.py" \
  --headless --enable_cameras \
  --experience "$ROOT/IsaacLab-Tactile/apps/isaacsim_4_5/isaaclab.python.headless.rendering.nongx.kit" \
  --policy-path "$POLICY" --dataset-root "$DATA" \
  --repo-id Dleshers/factory-peg-insert-causal-recovery-v2-original \
  --output "$ROOT/persistent/evaluation_results/visual.json" \
  --episodes 10 --seed 4100 --max-steps 360 --torque-mode none --n-action-steps 1
```

For a fair four-arm result use the supplied `run_factory_causal_recovery_v2_eval_after_training.sh`; it waits for all training jobs, then evaluates each arm with 10 identical seed values and stores JSON under `$ROOT/persistent/evaluation_results/`.

## Monitoring and recovery

```bash
tail -f "$ROOT/persistent/logs/factory_causal_recovery_v2_pipeline_20260724.log"
ps -eo pid,ppid,etime,stat,%cpu,cmd | grep -E '[i]saaclab|[l]erobot-train'
df -h /root/autodl-tmp
```

On an error, preserve the log and HDF5/checkpoint directory. Do not delete a partially created dataset or overwrite a run name; use a new run suffix after diagnosis. Runtime datasets/checkpoints/logs are excluded from Git. Publish only reviewed small metadata/result files to Hugging Face or GitHub.
