import inspect
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import ForceLSTMEncoder, SmolVLAPolicy, _reduce_action_losses

def test_smolvla_uses_lerobot_action_padding_key():
    source = inspect.getsource(SmolVLAPolicy.forward)
    assert 'batch.get("action_is_pad")' in source
    assert 'actions_id_pad' not in source



def test_force_lstm_encoder_shape_and_gradient():
    encoder = ForceLSTMEncoder(input_dim=1, hidden_dim=32, output_dim=16, num_layers=2)
    latent = encoder(torch.randn(4, 30, 1))
    latent.square().mean().backward()
    assert latent.shape == (4, 16)
    assert encoder.lstm.weight_ih_l0.grad is not None
    assert encoder.fc.weight.grad is not None


def test_torque_lstm_feature_shape_validation():
    config = SmolVLAConfig(use_torque_lstm=True)
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


def test_action_loss_padding_does_not_dilute_short_chunks():
    losses = torch.tensor([[[1.0, 1.0], [99.0, 99.0], [99.0, 99.0]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
    action_is_pad = torch.tensor([[False, True, True], [False, False, False]])
    per_sample = _reduce_action_losses(losses, action_is_pad, first_step_weight=1.0, reduction="none")
    torch.testing.assert_close(per_sample, torch.ones(2))


def test_action_loss_first_step_weight_matches_weighted_mean():
    losses = torch.tensor([[[4.0], [1.0], [1.0]]])
    action_is_pad = torch.tensor([[False, False, False]])
    result = _reduce_action_losses(losses, action_is_pad, first_step_weight=5.0)
    torch.testing.assert_close(result, torch.tensor(22.0 / 7.0))


def test_action_loss_all_padding_is_finite_zero():
    losses = torch.ones((1, 3, 2))
    action_is_pad = torch.ones((1, 3), dtype=torch.bool)
    result = _reduce_action_losses(losses, action_is_pad, first_step_weight=5.0)
    torch.testing.assert_close(result, torch.tensor(0.0))


def test_action_loss_first_step_weight_must_be_positive():
    with pytest.raises(ValueError, match="action_loss_first_step_weight"):
        SmolVLAConfig(action_loss_first_step_weight=0.0)
