# A100 training from Hugging Face raw data

## Scope and links

- Code revision: GitHub `Dleshers/fengyao_simulation_of_smolvla`, commit
  `e5dcdb2` or a newer compatible commit.
- Raw transfer dataset (private):
  <https://huggingface.co/datasets/Dleshers/factory-peg-insert-conditional-recovery-v3-raw>
- Expected HDF5 file inside the dataset:
  `peg_insert_demos.hdf5`.

The dataset is uploaded only after the 400-demo collector has passed its raw
audit. An authorized A100 machine must authenticate to Hugging Face as an
account permitted to read the private repository.

## Experimental contract

The data has 400 strict-success demonstrations:

| Type | Count | Measured initial XY error |
|---|---:|---|
| nominal insertion | 100 | n/a |
| recovery/easy | 100 | `[2.5, 4.5)` mm |
| recovery/medium | 100 | `[4.5, 6.0)` mm |
| recovery/hard | 100 | `[6.0, 7.5)` mm |

Recovery initializations are deterministic hand-IK plus grasped-peg state
interventions. They test a precisely controlled post-error recovery regime;
they must not be described as physical force-impulse experiments.

Every frame retains `joint_torque` as **signed 7D** data. The new policy uses a
causal `30 x 7` history and trains its torque LSTM from scratch. Do not supply
the historical one-channel `trained_lstm_weights/torque_16d_encoder.pt` to this
run.

## Download on the A100 machine

```bash
git clone https://github.com/Dleshers/fengyao_simulation_of_smolvla.git
cd fengyao_simulation_of_smolvla
git checkout e5dcdb2

ROOT="$PWD/_runtime/remote_handoff_gripper_lstm_work"
RUN=factory_peg_insert_conditional_recovery_v3_formal400_20260730
hf auth login
hf download Dleshers/factory-peg-insert-conditional-recovery-v3-raw \
  --repo-type dataset \
  --local-dir "$ROOT/persistent/raw_hdf5/$RUN"
```

The raw transfer is expected to be approximately 0.7--1.0 GiB. It is safer and
more resumable than GitHub for the HDF5 artifact. Keep the raw directory layout
unchanged, so the file resolves to:

```text
$ROOT/persistent/raw_hdf5/factory_peg_insert_conditional_recovery_v3_formal400_20260730/peg_insert_demos.hdf5
```

## Environment required for training

Install the repository's LeRobot/SmolVLA environment under
`$ROOT/.venv/lerobot`, make the official SmolVLA base checkpoint available at
`$ROOT/pretrained/official_smolvla_base`, and verify:

```bash
test -x "$ROOT/.venv/lerobot/bin/lerobot-train"
test -f "$ROOT/pretrained/official_smolvla_base/model.safetensors"
nvidia-smi
```

Isaac Sim is not required for conversion or training. It is only needed for
the final closed-loop evaluation stage.

## Convert and train four matched controls

Run the checked-in pipeline with collection skipped:

```bash
RUNTIME_ROOT="$ROOT" SKIP_COLLECTION=1 SKIP_EVALUATION=1 \
  bash experiment/run_factory_conditional_recovery_v3_pipeline.sh \
  2>&1 | tee "$ROOT/persistent/logs/a100_conditional_recovery_v3_train.log"
```

It first audits the raw HDF5, then creates three local datasets:

| Dataset suffix | Torque feature |
|---|---|
| `7d-original` | aligned signed 7D torque |
| `7d-zero` | all-zero 7D control |
| `7d-shuffle` | within-episode shuffled 7D control |

Recovery episodes are materialized twice during conversion. The pipeline then
trains sequentially for 50,000 steps: `visual`, `torque`, `zero`, `shuffle`.
All torque-bearing arms have identical 7D LSTM capacity and train that encoder;
only the torque input differs.

## Evaluation

If the A100 host also has a working Isaac Sim headless RGB environment, omit
`SKIP_EVALUATION=1`. The pipeline will automatically evaluate each arm on 100
matched near-rim starts (34 easy, 33 medium, 33 hard) and write:

```text
$ROOT/persistent/evaluation_results/conditional_recovery_v3_400demos_7d_20260730_r1/REPORT.md
```

Otherwise transfer the four A100 checkpoints back to the validated simulator
host and run the same pipeline's evaluation section there. Preserve policy
configurations and dataset metadata with the checkpoints.
