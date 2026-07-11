#!/usr/bin/env python3
"""Audit SmolVLA torque-token compatibility before training/evaluation.

This is a read-only guardrail script.  It checks the dataset schema, visual
baseline checkpoint, torque checkpoint config, and optional safetensors weights
for the controlled "visual + one gripper-torque LSTM suffix token" experiment.
It intentionally does not load the full SmolVLA model or start evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_TORQUE_ARCH = {
    "torque_window_size": 30,
    "torque_input_dim": 1,
    "torque_lstm_hidden_dim": 32,
    "torque_lstm_output_dim": 16,
    "torque_lstm_num_layers": 1,
}


@dataclass
class Finding:
    level: str
    message: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shape_of(features: dict[str, Any], key: str) -> tuple[int, ...] | None:
    feature = features.get(key)
    if feature is None:
        return None
    shape = feature.get("shape")
    return tuple(shape) if shape is not None else None


def type_of(features: dict[str, Any], key: str) -> str | None:
    feature = features.get(key)
    if feature is None:
        return None
    return feature.get("type") or feature.get("dtype")


def add_shape_check(findings: list[Finding], label: str, actual: tuple[int, ...] | None, expected: tuple[int, ...]) -> None:
    if actual == expected:
        findings.append(Finding("PASS", f"{label} shape is {actual}"))
    else:
        findings.append(Finding("FAIL", f"{label} shape is {actual}, expected {expected}"))


def audit_dataset(findings: list[Finding], dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    info = load_json(info_path)
    stats = load_json(stats_path) if stats_path.exists() else {}
    features = info.get("features", {})

    findings.append(Finding("INFO", f"dataset={dataset_root}"))
    findings.append(
        Finding(
            "INFO",
            f"episodes={info.get('total_episodes')} frames={info.get('total_frames')} fps={info.get('fps')}",
        )
    )
    add_shape_check(findings, "dataset observation.state", shape_of(features, "observation.state"), (9,))
    add_shape_check(findings, "dataset action", shape_of(features, "action"), (8,))
    add_shape_check(findings, "dataset camera1", shape_of(features, "observation.images.camera1"), (3, 224, 224))
    add_shape_check(findings, "dataset camera2", shape_of(features, "observation.images.camera2"), (3, 224, 224))
    add_shape_check(findings, "dataset gripper torque", shape_of(features, "observation.gripper_torque"), (30, 1))

    torque_stats = stats.get("observation.gripper_torque")
    if torque_stats:
        mean = torque_stats.get("mean", [None])[0]
        std = torque_stats.get("std", [None])[0]
        q50 = torque_stats.get("q50", [None])[0]
        q99 = torque_stats.get("q99", [None])[0]
        findings.append(Finding("INFO", f"raw torque stats mean={mean:.6g} std={std:.6g} q50={q50:.6g} q99={q99:.6g}"))
        if std is None or std < 1e-6:
            findings.append(Finding("FAIL", "gripper torque appears constant or has near-zero std"))
    else:
        findings.append(Finding("WARN", "dataset stats missing observation.gripper_torque"))
    return info


def audit_config_pair(
    findings: list[Finding],
    visual_config: dict[str, Any],
    torque_config: dict[str, Any],
) -> None:
    visual_inputs = visual_config.get("input_features", {})
    torque_inputs = torque_config.get("input_features", {})
    visual_outputs = visual_config.get("output_features", {})
    torque_outputs = torque_config.get("output_features", {})

    if "observation.gripper_torque" in visual_inputs:
        findings.append(Finding("FAIL", "visual baseline declares observation.gripper_torque"))
    else:
        findings.append(Finding("PASS", "visual baseline does not declare torque input"))

    shared_keys = ["observation.state", "observation.images.camera1", "observation.images.camera2"]
    for key in shared_keys:
        if visual_inputs.get(key) == torque_inputs.get(key):
            findings.append(Finding("PASS", f"shared input feature matches: {key}"))
        else:
            findings.append(Finding("FAIL", f"shared input feature differs: {key}"))

    if visual_outputs.get("action") == torque_outputs.get("action"):
        findings.append(Finding("PASS", "action output feature matches between visual and torque policies"))
    else:
        findings.append(Finding("FAIL", "action output feature differs between visual and torque policies"))

    if torque_config.get("use_torque_lstm") is True:
        findings.append(Finding("PASS", "torque policy enables use_torque_lstm"))
    else:
        findings.append(Finding("FAIL", "torque policy does not enable use_torque_lstm"))

    for key, expected in EXPECTED_TORQUE_ARCH.items():
        actual = torque_config.get(key)
        if actual == expected:
            findings.append(Finding("PASS", f"{key}={actual}"))
        else:
            findings.append(Finding("FAIL", f"{key}={actual}, expected {expected}"))

    if torque_config.get("train_torque_lstm") is False:
        findings.append(Finding("PASS", "train_torque_lstm=false: external encoder frozen by config"))
    else:
        findings.append(Finding("FAIL", "train_torque_lstm is not false"))

    if torque_config.get("torque_zero_init_adapter") is True:
        findings.append(Finding("PASS", "torque_zero_init_adapter=true"))
    else:
        findings.append(Finding("WARN", "torque_zero_init_adapter is not true"))

    gate = torque_config.get("torque_gate_init")
    if gate is None:
        findings.append(Finding("WARN", "torque_gate_init is absent; torque token is ungated"))
    elif isinstance(gate, (int, float)) and abs(float(gate)) <= 1.0:
        findings.append(Finding("PASS", f"torque_gate_init={gate}"))
    else:
        findings.append(Finding("WARN", f"torque_gate_init={gate}; verify this is intentional"))


def audit_preprocessor(findings: list[Finding], torque_policy: Path) -> None:
    preprocessor_path = torque_policy / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        findings.append(Finding("WARN", f"missing {preprocessor_path}"))
        return
    preprocessor = load_json(preprocessor_path)
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") != "normalizer_processor":
            continue
        config = step.get("config", {})
        features = config.get("features", {})
        norm_map = config.get("norm_map", {})
        torque_type = type_of(features, "observation.gripper_torque")
        torque_norm = norm_map.get(torque_type) if torque_type is not None else None
        if torque_type is None:
            findings.append(Finding("FAIL", "normalizer does not declare observation.gripper_torque"))
        else:
            findings.append(Finding("INFO", f"normalizer torque feature type={torque_type} norm={torque_norm}"))
        if torque_norm and torque_norm != "IDENTITY":
            findings.append(
                Finding(
                    "WARN",
                    "torque window is normalized before the frozen LSTM; this is train/eval-consistent, "
                    "but only physically correct if the external LSTM was intended to consume normalized torque",
                )
            )
        elif torque_norm == "IDENTITY" or torque_norm is None:
            findings.append(Finding("PASS", "torque window is not normalized by the policy preprocessor"))
        return
    findings.append(Finding("WARN", "policy_preprocessor has no normalizer_processor"))


def audit_weights(findings: list[Finding], torque_policy: Path) -> None:
    model_path = torque_policy / "model.safetensors"
    if not model_path.exists():
        findings.append(Finding("WARN", f"missing {model_path}; skipped weight-level audit"))
        return
    try:
        from safetensors.torch import load_file
    except Exception as exc:  # pragma: no cover - dependency may be absent
        findings.append(Finding("WARN", f"could not import safetensors for weight audit: {exc}"))
        return
    tensors = load_file(model_path)
    shape_expectations = {
        "model.torque_lstm.lstm.weight_ih_l0": (128, 1),
        "model.torque_lstm.lstm.weight_hh_l0": (128, 32),
        "model.torque_lstm.fc.weight": (16, 32),
        "model.torque_lstm.fc.bias": (16,),
    }
    for key, expected in shape_expectations.items():
        tensor = tensors.get(key)
        actual = None if tensor is None else tuple(tensor.shape)
        if actual == expected:
            findings.append(Finding("PASS", f"{key} shape={actual}"))
        else:
            findings.append(Finding("FAIL", f"{key} shape={actual}, expected {expected}"))

    adapter_w = tensors.get("model.torque_to_expert.weight")
    adapter_b = tensors.get("model.torque_to_expert.bias")
    if adapter_w is not None:
        findings.append(Finding("INFO", f"torque_to_expert.weight abs_max={float(adapter_w.abs().max()):.6g}"))
    if adapter_b is not None:
        findings.append(Finding("INFO", f"torque_to_expert.bias abs_max={float(adapter_b.abs().max()):.6g}"))
    gate = tensors.get("model.torque_gate")
    if gate is not None:
        findings.append(Finding("INFO", f"learned torque_gate={float(gate.item()):.6g}"))


def audit_task_mode(findings: list[Finding], task_mode: str, dataset_info: dict[str, Any]) -> None:
    features = dataset_info.get("features", {})
    state_shape = shape_of(features, "observation.state")
    action_shape = shape_of(features, "action")
    if task_mode == "pick_place_joint":
        if state_shape == (9,) and action_shape == (8,):
            findings.append(Finding("PASS", "pick_place_joint schema matches current SmolVLA checkpoints"))
        else:
            findings.append(Finding("FAIL", f"pick_place_joint expected state [9], action [8], got {state_shape}, {action_shape}"))
    elif task_mode == "peg_insert_ik":
        if state_shape == (49,) and action_shape == (7,):
            findings.append(Finding("PASS", "peg_insert_ik dataset schema matches Isaac peg-insert env"))
        else:
            findings.append(
                Finding(
                    "WARN",
                    "peg_insert_ik env smoke reports policy state [49] and action [7]; this dataset/checkpoint "
                    f"reports state {state_shape}, action {action_shape}. Do not reuse pick-place checkpoint/converter directly.",
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--visual-policy", type=Path, required=True)
    parser.add_argument("--torque-policy", type=Path, required=True)
    parser.add_argument("--task-mode", choices=("pick_place_joint", "peg_insert_ik"), default="pick_place_joint")
    args = parser.parse_args()

    findings: list[Finding] = []
    dataset_info = audit_dataset(findings, args.dataset_root)
    visual_config = load_json(args.visual_policy / "config.json")
    torque_config = load_json(args.torque_policy / "config.json")
    audit_config_pair(findings, visual_config, torque_config)
    audit_preprocessor(findings, args.torque_policy)
    audit_weights(findings, args.torque_policy)
    audit_task_mode(findings, args.task_mode, dataset_info)

    worst = "PASS"
    for finding in findings:
        print(f"{finding.level}: {finding.message}")
        if finding.level == "FAIL":
            worst = "FAIL"
        elif finding.level == "WARN" and worst != "FAIL":
            worst = "WARN"
    print(f"SUMMARY: {worst}")
    return 2 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
