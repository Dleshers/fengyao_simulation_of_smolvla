import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import ForceLSTMEncoder


def test_force_lstm_encoder_shape_and_gradient():
    encoder = ForceLSTMEncoder(input_dim=1, hidden_dim=32, output_dim=16, num_layers=2)
    torque = torch.randn(4, 30, 1)

    latent = encoder(torque)
    latent.square().mean().backward()

    assert latent.shape == (4, 16)
    assert encoder.lstm.weight_ih_l0.grad is not None
    assert encoder.fc.weight.grad is not None


def test_torque_lstm_feature_shape_validation():
    config = SmolVLAConfig(
        use_torque_lstm=True,
        torque_window_key="observation.gripper_torque",
        torque_window_size=30,
        torque_input_dim=1,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
        "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.gripper_torque": PolicyFeature(type=FeatureType.ENV, shape=(30, 2)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))}

    with pytest.raises(ValueError, match="must have shape"):
        config.validate_features()


def test_visual_baseline_does_not_require_torque_feature():
    config = SmolVLAConfig(use_torque_lstm=False)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
        "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))}

    config.validate_features()
