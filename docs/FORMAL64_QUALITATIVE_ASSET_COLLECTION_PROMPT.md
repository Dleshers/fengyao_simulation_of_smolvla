# Formal64 Qualitative Asset Collection Request

## Purpose

Prepare auditable qualitative visual material for the UCL MSc thesis on torque-conditioned SmolVLA control in the Isaac Sim/Isaac Lab Factory peg-in-hole task. The material must support, rather than replace, the reported Formal64 quantitative evaluation.

Only the following three conditions are in scope:

- `visual`: visual-only policy;
- `torque-original`: policy conditioned on the recorded causal torque-history window;
- `torque-zero`: policy using a zero torque-history input.

Do **not** collect, render, or discuss shuffle-condition material for the thesis.

## Experimental integrity requirements

1. Reuse the frozen Formal64 checkpoint, evaluation configuration, recorded initial snapshots, seeds, and decision rules.
2. Do not change the policy, action scaling, controller, termination thresholds, randomisation, or task difficulty to obtain better-looking scenes.
3. All paired comparisons must replay the same recorded initial snapshot for each condition.
4. Where the requested asset is unavailable, report this explicitly instead of substituting an unpaired or newly generated state.
5. Preserve the existing result audit: record selected `pair_id`, checkpoint, seed, condition, frame/step, and outcome labels for every exported image.

The official Formal64 evaluation contains 64 same-snapshot pairs. The reported aggregate strict-success rates are 68.8% (44/64) for `visual`, 75.0% (48/64) for `torque-zero`, and 87.5% (56/64) for `torque-original`.

## Required assets

### 1. Task and observation overview

Export a clean task overview showing the robot, peg, insertion hole, and the near-contact geometry.

- Views: one oblique overview and one side view.
- If feasible, also show the viewpoints corresponding to the two policy cameras.
- Suggested paths:
  - `formal64_qualitative/task_overview/scene_overview_oblique.png`
  - `formal64_qualitative/task_overview/scene_overview_side.png`
- Resolution: at least 1280 x 720 for thesis illustrations.
- No GUI, debug, oracle, or success/failure overlay in the exported image.

### 2. Two paired original-only-success cases

Select two `pair_id` values for which:

- `torque-original` is a strict success; and
- `visual` is a strict failure.

Prefer one low-load and one high-load case, with different direction sectors where the official records permit this. For each selected pair, export matched frames from both conditions using the same recorded initial snapshot:

1. initial near-contact state;
2. lateral recovery/alignment stage;
3. terminal outcome.

For each stage, export the raw policy observations from camera 1 and camera 2. If high-resolution illustrative renders are produced, keep them separate from the raw observations and state that they are illustrative render products rather than policy inputs.

Suggested naming convention:

```text
formal64_qualitative/paired_original_only_success/<pair_id>/
  visual_camera1_initial.png
  visual_camera2_initial.png
  visual_camera1_recovery.png
  visual_camera2_recovery.png
  visual_camera1_terminal.png
  visual_camera2_terminal.png
  torque_original_camera1_initial.png
  torque_original_camera2_initial.png
  torque_original_camera1_recovery.png
  torque_original_camera2_recovery.png
  torque_original_camera1_terminal.png
  torque_original_camera2_terminal.png
  metadata.json
```

### 3. One original-policy ejection safety case

Select one official `pair_id` in which `torque-original` triggers ejection. Prefer a pair where `visual` or `torque-zero` does not eject, if such a matched comparator exists.

Export:

1. initial near-contact state;
2. immediately before ejection;
3. the ejection/termination state;
4. matched frames for the non-ejection comparator, where available.

This case must be described as a safety/failure analysis example, not as evidence against the aggregate success result. In Formal64, ejection counts were 3 for `torque-original`, 1 for `visual`, and 0 for `torque-zero`.

### 4. Synchronized quantitative traces

For every selected case, save a synchronized trace file (`.npz` or `.csv`) containing, at minimum:

- 7-D torque-history input/window;
- peg XY error;
- insertion depth;
- executed action;
- strict-success, ejection, pass-through, and drift flags.

These traces will support figure captions and ensure that the visual interpretation remains linked to recorded behaviour.

### 5. Required metadata

Each selected case must include `metadata.json` with the following fields:

```json
{
  "pair_id": "...",
  "branch": "visual | torque-original | torque-zero",
  "checkpoint": "...",
  "seed": "...",
  "load_band": "low | high | unknown",
  "direction_sector": "...",
  "frame_or_step": 0,
  "camera_name": "camera1 | camera2 | illustrative",
  "strict_success": false,
  "ejection": false,
  "pass_through": false,
  "min_xy_error": 0.0,
  "final_depth": 0.0
}
```

## Rendering requirements

- Use consistent viewpoints within a paired comparison.
- Export lossless PNG images; do not use screenshots of an application window.
- Keep raw policy observation images distinct from high-resolution illustrative renders.
- Do not include debug lines, oracle state, manually edited annotations, or outcome text over the image. Annotations will be added separately in the thesis figure layout.
- Report absolute output paths and a compact mapping from every selected image to its `pair_id`, condition, and frame/step.

## Headless rendering guidance

Headless capture is feasible for this task. The existing paired evaluator already constructs Replicator cameras and render products, calls the simulator render step, and reads RGB observations. It currently uses two policy camera viewpoints and low-resolution render products for policy inference.

For the thesis assets:

1. Run the existing Isaac Lab/Isaac Sim evaluation launcher in its supported headless mode with camera rendering enabled. The exact command should follow the server's installed launcher and version; do not invent a separate simulator configuration.
2. Call the simulator render step before reading the RGB annotator output, then write the returned RGB arrays directly to PNG.
3. Preserve the existing low-resolution policy render products unchanged for evaluation reproducibility.
4. Create a separate high-resolution render product (at least 1280 x 720) only for illustrative images. It must observe the same replayed scene and must not alter policy inputs, action execution, or outcome rules.
5. The server needs an RTX-capable GPU and a functioning headless graphics backend (for example EGL/Vulkan as supported by the installed Isaac Sim version). A desktop display or X window is not required when the configured headless backend is available.
6. For same-snapshot Formal64 figures, extract frames from the official saved/replayed snapshot rather than from a newly sampled simulation state. Confirm that the first policy RGB observation still matches the recorded evaluation audit where that check is available.

## Delivery checklist

Return the following after collection:

1. A table mapping each exported asset to `pair_id`, condition, load band, direction sector, frame/step, and official outcome.
2. Absolute paths to all PNG images, metadata files, and trace files.
3. Confirmation that each paired comparison used the same official initial snapshot.
4. The launcher command and relevant rendering flags actually used on the server.
5. A list of any unavailable requested cases or assets, with the reason; do not fabricate replacements.
