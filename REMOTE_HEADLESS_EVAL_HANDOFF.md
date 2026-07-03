# Remote headless handoff: SmolVLA visual vs gripper-torque LSTM

> Audience: the AI agent operating the remote GPU machine. Read this document completely before changing code.

## 1. Objective and current state

The experiment compares:

1. visual-only SmolVLA;
2. the same visual/state policy plus a causal gripper-torque window, encoded by an LSTM and injected into the Action Expert suffix.

Authoritative tactile contract:

```text
observation.gripper_torque: float32 [batch, 30, 1]
oldest sample -> newest sample
LSTM: input=1, hidden=32, layers=1, output=16
action: float32 [batch, horizon, 8]
```

The remote machine has no desktop. That is supported: Isaac Sim must run with `--headless`, but cameras must remain enabled. Do not disable rendering/cameras just because no window is displayed.

Available checkpoints on the source machine:

```text
visual, 50000 steps:
/home/radu/SmolVLA-Fengyao/training_runs/exp01__visual__steps50000__seed1000/checkpoints/050000/pretrained_model

visual + torque LSTM, 30000 steps:
/home/radu/SmolVLA-Fengyao/training_runs/20260703_104740__torque_lstm_pretrained_h32_l1__steps50000__seed1000/checkpoints/030000/pretrained_model
```

The torque run reached step 37033, but the source GPU driver failed before the next checkpoint. Step 30000 is the newest complete and verified checkpoint. It includes the trained torque LSTM inside `model.safetensors`; the original standalone LSTM file is not required for inference.

Important fairness caveat: the visual 50000 checkpoint predates the feature-schema refresh fix. Its saved config contains the inherited base schema (6D state and three camera slots), whereas the torque checkpoint contains the actual 9D state and two 224x224 cameras. Use the existing checkpoints for deployment smoke tests and exploratory rollouts. Do not report them as the final controlled comparison without retraining or validating the visual baseline with the same two-camera/9D schema.

## 2. Files to obtain from Git

The handoff repository must contain this directory intact:

```text
remote_handoff_gripper_lstm/
  README_FOR_REMOTE_AGENT.md
  instructions/
  lerobot_overrides/
  isaaclab_patches/
  references/
```

Also obtain these current simulation files:

```text
simulation/REMOTE_HEADLESS_EVAL_HANDOFF.md
simulation/install_gripper_lstm_overrides.sh
simulation/SETUP_TORQUE_WINDOW_PATCHES.md
simulation/activate_simulation_env.sh
```

Read `remote_handoff_gripper_lstm/README_FOR_REMOTE_AGENT.md` next. The files under `lerobot_overrides/` are authoritative. They include two fixes added during training:

- accept LeRobot's `[B,1,30,1]` sampled batch and reduce it to `[B,30,1]`;
- use a safetensors-compatible LSTM whose CUDA parameters are not rewired into an unsavable shared storage.

The LeRobot policy factory must also refresh `cfg.input_features` and `cfg.output_features` from the active dataset/checkpoint interface. A pretrained base config must not suppress `observation.gripper_torque`.

## 3. Checkpoint transfer (not ordinary Git)

Each model is about 907 MB. Do not commit these blobs to ordinary Git history. Transfer the two `pretrained_model` directories with `rsync`/`scp`, or use Git LFS only if the selected Git server explicitly supports the required quota.

For this handoff, both directories are also published as assets of GitHub Release `checkpoints-v1`. On the remote machine:

```bash
gh release download checkpoints-v1 \
  --repo Dleshers/fengyao_simulation_of_smolvla \
  --pattern '*.tar.gz' \
  --dir ~/smolvla_eval/downloads

(cd ~/smolvla_eval/downloads && sha256sum -c \
  PATH_TO_CLONE/RELEASE_ASSETS.sha256)

mkdir -p ~/smolvla_eval/checkpoints
tar -xzf ~/smolvla_eval/downloads/visual_050000.tar.gz \
  -C ~/smolvla_eval/checkpoints
tar -xzf ~/smolvla_eval/downloads/torque_lstm_030000.tar.gz \
  -C ~/smolvla_eval/checkpoints
```

Example from the source machine:

```bash
rsync -aP \
  /home/radu/SmolVLA-Fengyao/training_runs/exp01__visual__steps50000__seed1000/checkpoints/050000/pretrained_model/ \
  REMOTE_USER@REMOTE_HOST:~/smolvla_eval/checkpoints/visual_050000/

rsync -aP \
  /home/radu/SmolVLA-Fengyao/training_runs/20260703_104740__torque_lstm_pretrained_h32_l1__steps50000__seed1000/checkpoints/030000/pretrained_model/ \
  REMOTE_USER@REMOTE_HOST:~/smolvla_eval/checkpoints/torque_lstm_030000/
```

Required SHA-256 checks:

```text
visual model.safetensors:
bf38641acf764526603372f32d2bdc4028cee2b36be523e6facd3f016aea08ea

torque model.safetensors:
8636e40daa38af4bd5a9ee6ce573dd43c05745ad2a1f989ae037aff2f7446fa0
```

Verify remotely:

```bash
sha256sum \
  ~/smolvla_eval/checkpoints/visual_050000/model.safetensors \
  ~/smolvla_eval/checkpoints/torque_lstm_030000/model.safetensors
```

## 4. Remote installation

Keep large runtimes, Isaac assets, caches and checkpoints on the remote machine's large local Linux filesystem. Do not place Conda environments or active Isaac caches on NTFS/network storage.

Use compatible versions unless the remote agent deliberately ports and retests the patches:

```text
Isaac Sim 4.5.0
IsaacLab-Tactile matching the supplied patch
Python 3.10 for IsaacLab
Python 3.12 LeRobot environment used by the policy
torch 2.7.x CUDA build
pyzmq
```

Install the LeRobot overrides into the remote LeRobot clone and apply the supplied IsaacLab patch:

```bash
cp remote_handoff_gripper_lstm/lerobot_overrides/configuration_smolvla.py \
  LEROBOT/src/lerobot/policies/smolvla/configuration_smolvla.py
cp remote_handoff_gripper_lstm/lerobot_overrides/modeling_smolvla.py \
  LEROBOT/src/lerobot/policies/smolvla/modeling_smolvla.py

cd ISAACLAB_TACTILE
git apply --check ../remote_handoff_gripper_lstm/isaaclab_patches/SETUP_torque_window_gripper_IsaacLab.patch
git apply ../remote_handoff_gripper_lstm/isaaclab_patches/SETUP_torque_window_gripper_IsaacLab.patch
```

Do not apply the obsolete flattened-linear LeRobot torque patch. Do not substitute the full-body `[30,9]` torque path for this experiment.

Before evaluation, verify CUDA and headless rendering:

```bash
nvidia-smi
python -c "import torch; torch.cuda.init(); print(torch.cuda.get_device_name(0))"
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

## 5. Headless server contract

Start IsaacLab as a headless ZeroMQ server. The exact script flags may differ slightly after patch application; inspect `python scripts/eval_server.py --help` and preserve these semantics:

```bash
cd ISAACLAB_TACTILE
./isaaclab.sh -p scripts/eval_server.py \
  --headless \
  --enable_cameras \
  --host '*' \
  --port 5555 \
  --env Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0 \
  --send-gripper-torque-window \
  --torque-window-size 30
```

The server must expose two RGB cameras, 9D state, and the mean of the two gripper-joint torques as a causal `[30,1]` window. Reset must clear the history. Left padding must repeat the first valid value, matching dataset conversion.

Since there is no GUI, save rollout evidence server-side: episode success, step count, actions, gripper torque, object/goal poses, and optionally camera frames or MP4. Headless does not mean unobservable.

## 6. Policy client and staged validation

Run the policy client in a second process/environment on the same machine and connect to `127.0.0.1:5555`. First perform only reset plus one zero/safe action. Print and assert shapes before loading a policy:

```text
camera1/camera2: [1,3,224,224] after preprocessing
state:           [1,9]
gripper torque:  [1,30,1]
action:          [1,8]
```

Then perform one deterministic episode with each checkpoint, followed by at least 20 matched episodes. Use the same simulator seed sequence, initial states, task text, horizon and success criterion for both policies.

For the torque checkpoint, the client configuration must preserve:

```text
use_torque_lstm=true
torque_window_key=observation.gripper_torque
torque_window_size=30
torque_input_dim=1
torque_lstm_hidden_dim=32
torque_lstm_output_dim=16
torque_lstm_num_layers=1
```

Do not pass the standalone LSTM weights during checkpoint inference; doing so risks overwriting the task-trained LSTM stored in the checkpoint.

## 7. Acceptance criteria

The remote agent should report:

- Git commit SHA used for the handoff;
- LeRobot and IsaacLab commit SHAs;
- checkpoint SHA-256 values;
- CUDA/driver/GPU versions;
- exact server and client commands;
- observation/action shape dump;
- per-seed success and aggregate success rate;
- mean episode length and action smoothness;
- peak joint velocity/acceleration, collisions and safety violations;
- saved headless rollout videos or frame archives where possible;
- every local code change made to port the handoff.

Stop immediately on NaN/Inf actions, observation shape drift, missing torque, persistent zero torque, violent motion, or a policy/server action-dimension mismatch.

## 8. Optional continuation of torque training

If desired, transfer the complete checkpoint directory (including `training_state/`, about 1.3 GB), not only `pretrained_model/`, and resume from step 30000. Do not claim the model is a 50000-step model until a complete 50000 checkpoint exists.
