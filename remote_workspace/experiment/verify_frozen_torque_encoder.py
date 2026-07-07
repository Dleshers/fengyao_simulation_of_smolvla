#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.policies.smolvla.modeling_smolvla import ForceLSTMEncoder


EXPECTED_SHAPES = {
    "lstm.weight_ih_l0": (128, 1),
    "lstm.weight_hh_l0": (128, 32),
    "lstm.bias_ih_l0": (128,),
    "lstm.bias_hh_l0": (128,),
    "fc.weight": (16, 32),
    "fc.bias": (16,),
}


def extract_encoder_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.jit.load(str(path), map_location="cpu")
    state: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.state_dict().items():
        key = key.removeprefix("full_model.")
        key = key.replace("encoder_lstm.", "lstm.").replace("encoder_fc.", "fc.")
        if key.startswith(("lstm.", "fc.")):
            state[key] = value
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()

    state = extract_encoder_state(args.weights)
    actual_shapes = {key: tuple(value.shape) for key, value in state.items()}
    if actual_shapes != EXPECTED_SHAPES:
        raise ValueError(f"Unexpected torque encoder tensors: {actual_shapes}")

    encoder = ForceLSTMEncoder(input_dim=1, hidden_dim=32, output_dim=16, num_layers=1)
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False)
    norm = torch.nn.LayerNorm(16)
    projection = torch.nn.Linear(16, 128)

    torque = torch.randn(4, 30, 1)
    latent = encoder(torque)
    token = projection(norm(latent)).unsqueeze(1)
    token.square().mean().backward()

    if tuple(latent.shape) != (4, 16) or tuple(token.shape) != (4, 1, 128):
        raise ValueError(f"Unexpected latent/token shapes: {latent.shape}, {token.shape}")
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise ValueError("Frozen torque encoder received gradients")
    if norm.weight.grad is None or projection.weight.grad is None:
        raise ValueError("Trainable normalization/projection did not receive gradients")
    if not torch.isfinite(token).all():
        raise ValueError("Torque token contains NaN or Inf")

    print("OK: strict external encoder load: input=1 hidden=32 layers=1 output=16")
    print("OK: encoder frozen; LayerNorm and Action Expert projection receive gradients")
    print("OK: [B,30,1] -> [B,16] -> [B,1,D_expert]")


if __name__ == "__main__":
    main()
