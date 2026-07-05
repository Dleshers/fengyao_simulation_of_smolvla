import pytest
import torch

from validate_dataset import audit_causal_torque_windows


class FakeDataset:
    def __init__(self, episodes):
        self.rows = []
        for episode_index, values in enumerate(episodes):
            history = []
            for value in values:
                history.append(float(value))
                padded = [history[0]] * (4 - len(history)) + history[-4:]
                self.rows.append(
                    {
                        "episode_index": torch.tensor(episode_index),
                        "observation.gripper_torque": torch.tensor(padded).reshape(4, 1),
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_accepts_causal_windows_and_episode_resets():
    dataset = FakeDataset([[1.0, 2.0, 3.0], [9.0, 8.0]])
    audit_causal_torque_windows(dataset, "observation.gripper_torque", 4, 10)


def test_rejects_cross_episode_or_noncausal_window():
    dataset = FakeDataset([[1.0, 2.0, 3.0]])
    dataset.rows[2]["observation.gripper_torque"][0, 0] = 99.0
    with pytest.raises(ValueError, match="not causal/contiguous"):
        audit_causal_torque_windows(dataset, "observation.gripper_torque", 4, 10)


def test_rejects_zero_padded_episode_start():
    dataset = FakeDataset([[5.0, 6.0]])
    dataset.rows[0]["observation.gripper_torque"][0, 0] = 0.0
    with pytest.raises(ValueError, match="incorrect padding"):
        audit_causal_torque_windows(dataset, "observation.gripper_torque", 4, 10)
