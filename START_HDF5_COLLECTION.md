# Start HDF5 Collection Immediately

This is the shortest safe path when the remote GPU machine becomes available.

## 1. Update the handoff repository

```bash
cd ~/fengyao_simulation_of_smolvla
git pull --ff-only origin main
```

If the clone is elsewhere, use that path in the commands below.

## 2. Restore patches and scripts into the runtime workspace

The existing LeRobot and IsaacLab clones must be present under the default work root:

```text
/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work/lerobot-tactile
/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work/IsaacLab-Tactile
```

Run:

```bash
cd ~/fengyao_simulation_of_smolvla
INSTALL_DEPS=1 bash restore_remote_workspace.sh
```

Use `INSTALL_DEPS=0` on later runs when the environments are already installed. The restore script
applies the LeRobot, policy-factory and IsaacLab patches, installs the authoritative policy files and
copies current experiment scripts into the runtime workspace.

## 3. Run preflight

```bash
cd /scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work
RUN_ISAAC_SMOKE=1 bash experiment/preflight_collection.sh
```

Do not collect until this prints `PASS`. It checks the GPU, CUDA, Python environments, storage,
required source changes and an optional minimal headless Isaac launch.

## 4. Collect the two-demo smoke HDF5

Use a persistent terminal so SSH disconnection does not kill Isaac Sim:

```bash
tmux new -s gripper-smoke
cd /scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work
bash experiment/run_hdf5_smoke.sh
```

Detach with `Ctrl-b d`. Reattach with:

```bash
tmux attach -t gripper-smoke
```

The script creates a timestamped directory under:

```text
/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/datasets/
```

It produces:

```text
data.hdf5
data.hdf5.sha256
collection.log
```

It also runs the raw-HDF5 schema/finite-value audit automatically. Review `collection.log` before
starting the full run.

## 5. Collect the full HDF5

After the smoke dataset passes:

```bash
tmux new -s gripper-full
cd /scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work
bash experiment/run_hdf5_full.sh
```

Defaults are:

```text
NUM_DEMOS=200
NUM_ENVS=4
SEED=1000
state=[9]
action=[8]
camera1/camera2=224x224 RGB
gripper_torque=[1] per raw frame
fps=20
```

Override deliberately, for example:

```bash
NUM_DEMOS=20 NUM_ENVS=2 SEED=1001 bash experiment/run_hdf5_full.sh
```

Never reuse an output directory containing `data.hdf5`; the scripts refuse to overwrite it.

## 6. Re-audit or convert an HDF5

Set `RAW_DIR` to the exact timestamped output directory printed by the collection script:

```bash
cd /scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work
export RAW_DIR=/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/datasets/EXACT_OUTPUT_DIR

.venv/lerobot/bin/python experiment/inspect_raw_hdf5.py "$RAW_DIR/data.hdf5"
sha256sum -c "$RAW_DIR/data.hdf5.sha256"
```

Convert only after the audit passes:

```bash
export DATASET_REPO_ID=franka_pickplace_joint_visual_torque_w30_v1
bash experiment/rebuild_dataset.sh convert
```

Conversion creates causal `observation.gripper_torque [30,1]` windows, repeats the first valid value
at episode startup, and then runs `validate_dataset.py`, including adjacent-window and episode-boundary
checks.

## Stop conditions

Stop and preserve the log if any of these occur:

- missing or black camera frames;
- state/action shape differs from `[9]`/`[8]`;
- NaN/Inf or constant gripper torque;
- HDF5 field lengths differ;
- repeated Isaac crashes or CUDA errors;
- causal-window validation fails after conversion.
