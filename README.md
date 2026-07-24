# Causal torque recovery benchmark for SmolVLA

This repository contains the reproducible code and runtime patches for a strict simulated Factory peg-in-hole benchmark. It compares visual SmolVLA with a frozen causal gripper-torque LSTM token, plus zero-torque and shuffled-torque controls.

The supported experiment is **Factory causal-recovery v2**:

- state: `float32[12]` = 9 robot joints + 3D fingertip midpoint;
- action: `float32[6]` Factory delta pose;
- observations: two RGB cameras (`224×224` after conversion);
- torque: causal scalar-L2 history `float32[30,1]`, oldest → newest;
- raw collection: observations are recorded **before** executing `action_t`;
- recovery data: an unlabelled lateral perturbation followed by oracle-labelled corrective actions;
- strict success: lateral error `< 2.5 mm` and relative insertion depth `< 1 mm`.

Start with [docs/REPRODUCE_FACTORY_CAUSAL_RECOVERY_V2.md](docs/REPRODUCE_FACTORY_CAUSAL_RECOVERY_V2.md). It is the only current operational guide.

Historical reports under `experiment_results/` are retained as evidence, not as instructions. Old handoff documents and legacy pick-and-place instructions were removed because their paths, schemas, and camera assumptions do not reproduce this benchmark.

Large files—Isaac Sim, source clones, datasets, checkpoints, logs, and Hugging Face caches—are intentionally excluded from Git. Runtime patches and an exact Python package snapshot are tracked here.
