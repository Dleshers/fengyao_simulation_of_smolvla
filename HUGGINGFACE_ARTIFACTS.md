# Hugging Face artifact handoff

This document is the authoritative remote-machine handoff for large artifacts that are intentionally excluded from Git history.

## Repositories and publication state

| Artifact | Hugging Face repository | State |
| --- | --- | --- |
| Validated LeRobot dataset | `Dleshers/franka-pickplace-joint-visual-torque-w30-v1` | Uploaded; private |
| Visual-only SmolVLA baseline | `Dleshers/smolvla-franka-pickplace-baseline-50k-seed1000` | Uploaded automatically only after the 50K run and final-checkpoint audit; private |
| Peg-insert torque-disambiguation diagnostic dataset | `Dleshers/peg-insert-torque-disambiguation-compact21-v1` | Uploaded; private |
| Peg-insert zero-torque diagnostic control | `Dleshers/peg-insert-torque-disambiguation-compact21-zero-v1` | Uploaded; private |
| Peg-insert global-shuffle torque diagnostic control | `Dleshers/peg-insert-torque-disambiguation-compact21-shuffleglobal-v1` | Uploaded; private |

Both repositories are private while redistribution terms for all simulation assets are reviewed. A remote agent needs a Hugging Face account with access to the `Dleshers` repositories and a read token. Never commit or print the token.

## Install and authenticate

Install the current Hugging Face CLI (`hf`, not the deprecated `huggingface-cli`):

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
source ~/.profile
hf auth login
hf auth whoami
```

Create a read token at <https://huggingface.co/settings/tokens>. Paste it only into the interactive `hf auth login` prompt.

For an ephemeral CI or agent environment, inject `HF_TOKEN` through its secret manager rather than placing it in a repository or shell script.

## Restore the dataset

From the restored workspace root:

```bash
ROOT="$PWD"
mkdir -p "$ROOT/datasets"
hf download Dleshers/franka-pickplace-joint-visual-torque-w30-v1 \
  --repo-type dataset \
  --local-dir "$ROOT/datasets/franka_pickplace_joint_visual_torque_w30_v1"
```

Expected dataset metadata:

- LeRobot codebase version: v3.0
- Episodes: 200
- Frames: 41,276
- Frequency: 20 Hz
- State: float32 `[9]`
- Action: float32 `[8]`
- RGB: two `[3,224,224]` cameras
- Torque history: float32 `[30,1]`, newest sample at index `-1`
- Source HDF5 SHA-256: `09a83546afa456efc1e0593dc431da8231b39eb455b9b725b477dacb30bad60f`

Perform a basic local check:

```bash
test -s "$ROOT/datasets/franka_pickplace_joint_visual_torque_w30_v1/meta/info.json"
python - <<'PY'
import json
from pathlib import Path

p = Path("datasets/franka_pickplace_joint_visual_torque_w30_v1/meta/info.json")
info = json.loads(p.read_text())
assert info["total_episodes"] == 200
assert info["total_frames"] == 41276
assert info["features"]["observation.state"]["shape"] == [9]
assert info["features"]["action"]["shape"] == [8]
assert info["features"]["observation.gripper_torque"]["shape"] == [30, 1]
print("Dataset metadata OK")
PY
```

After dependencies and workspace patches are installed, run the full project validator:

```bash
python remote_workspace/experiment/validate_dataset.py \
  --repo-id franka_pickplace_joint_visual_torque_w30_v1 \
  --root "$ROOT/datasets/franka_pickplace_joint_visual_torque_w30_v1" \
  --window-size 30 --samples 64 --sequence-checks 1024
```

## Restore the final baseline model

First check whether the audited 50K artifact is available:

```bash
hf models info Dleshers/smolvla-franka-pickplace-baseline-50k-seed1000
```

If it exists, download it:

```bash
mkdir -p "$ROOT/pretrained"
hf download Dleshers/smolvla-franka-pickplace-baseline-50k-seed1000 \
  --local-dir "$ROOT/pretrained/smolvla_franka_pickplace_baseline_50k_seed1000"
```

Verify the inference artifact:

```bash
MODEL="$ROOT/pretrained/smolvla_franka_pickplace_baseline_50k_seed1000"
test -s "$MODEL/model.safetensors"
test -s "$MODEL/config.json"
test -s "$MODEL/train_config.json"
test -s "$MODEL/policy_preprocessor.json"
test -s "$MODEL/policy_postprocessor.json"
```

The model repository contains only the final `pretrained_model` artifact. Intermediate checkpoints and optimizer state are deliberately omitted. It is a visual-only baseline: `use_torque_lstm=false` and tactile policy inputs are disabled.

## Restore the peg-insert torque-disambiguation diagnostic datasets

These three private LeRobot v3 datasets are diagnostic assets for validating whether the gated/zero-init torque-LSTM injection path can use gripper torque when the action label is intentionally ambiguous from vision/state alone. They are not final physical peg-insertion benchmark datasets.

Repositories:

```text
Dleshers/peg-insert-torque-disambiguation-compact21-v1
Dleshers/peg-insert-torque-disambiguation-compact21-zero-v1
Dleshers/peg-insert-torque-disambiguation-compact21-shuffleglobal-v1
```

Restore all three from the workspace root:

```bash
ROOT="$PWD"
mkdir -p "$ROOT/datasets"

hf download Dleshers/peg-insert-torque-disambiguation-compact21-v1 \
  --repo-type dataset \
  --local-dir "$ROOT/datasets/peg_insert_torque_disambiguation_compact21_v1"

hf download Dleshers/peg-insert-torque-disambiguation-compact21-zero-v1 \
  --repo-type dataset \
  --local-dir "$ROOT/datasets/peg_insert_torque_disambiguation_compact21_zero_v1"

hf download Dleshers/peg-insert-torque-disambiguation-compact21-shuffleglobal-v1 \
  --repo-type dataset \
  --local-dir "$ROOT/datasets/peg_insert_torque_disambiguation_compact21_shuffleglobal_v1"
```

Expected metadata for each dataset:

- Episodes: 45
- Frames: 1,755
- State: float32 `[21]`
- Action: float32 `[7]`
- RGB: two `[3,224,224]` cameras
- Torque history: float32 `[30,1]`

Control semantics:

- `compact21-v1`: correctly aligned diagnostic torque.
- `compact21-zero-v1`: torque windows are all zeros.
- `compact21-shuffleglobal-v1`: raw torque samples are globally permuted before causal windowing, preserving marginal torque scale while breaking contact-mode alignment.

Basic metadata check:

```bash
python - <<'PY'
import json
from pathlib import Path

roots = [
    Path("datasets/peg_insert_torque_disambiguation_compact21_v1"),
    Path("datasets/peg_insert_torque_disambiguation_compact21_zero_v1"),
    Path("datasets/peg_insert_torque_disambiguation_compact21_shuffleglobal_v1"),
]
for root in roots:
    info = json.loads((root / "meta/info.json").read_text())
    assert info["total_episodes"] == 45, root
    assert info["total_frames"] == 1755, root
    assert info["features"]["observation.state"]["shape"] == [21], root
    assert info["features"]["action"]["shape"] == [7], root
    assert info["features"]["observation.gripper_torque"]["shape"] == [30, 1], root
    print(f"OK: {root}")
PY
```

Reference experiment notes:

```text
experiment_results/2026-07-10_torque_disambiguation_dataset.md
```

The 300-step diagnostic checkpoints were intentionally not uploaded to Hugging Face because they are short smoke/diagnostic runs rather than final model artifacts.

## Agent rules

1. Treat the Hugging Face copies as immutable run artifacts. Do not overwrite them with a resumed or differently configured run.
2. Keep repository IDs and local paths distinct: Hub names use hyphens, while the established local dataset directory uses underscores.
3. Do not silently substitute the historical release checkpoints; they are not valid for the final controlled comparison.
4. Do not make any Hugging Face repository public until simulation-asset redistribution terms have been reviewed.
5. Record the exact Hub revision returned by `hf download` in any evaluation report.
