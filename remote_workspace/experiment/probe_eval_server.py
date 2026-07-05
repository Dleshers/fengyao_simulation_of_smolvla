#!/usr/bin/env python3
import argparse

import numpy as np

from lerobot.envs.isaaclab_tactile_remote import IsaacLabTactileRemoteEnv
from lerobot.processor.env_processor import IsaacLabTactilePolicyObservationProcessorStep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--control-mode", choices=("ik_rel", "joint"), default="joint")
    args = parser.parse_args()

    env = IsaacLabTactileRemoteEnv(
        server_host=args.host,
        server_port=args.port,
        include_gripper_torque_window=True,
        torque_window_size=30,
        control_mode=args.control_mode,
    )
    try:
        observation, _ = env.reset(seed=1000)
        torque = np.asarray(observation["gripper_torque"])
        if torque.dtype != np.float32 or torque.shape != (30, 1):
            raise ValueError(f"Expected float32 [30,1], got dtype={torque.dtype}, shape={torque.shape}")
        if not np.allclose(torque, torque[0]):
            raise ValueError("Reset torque window is not left-padded with the first valid sample")
        print(f"OK: gripper_torque dtype={torque.dtype} shape={torque.shape} newest={torque[-1, 0]:.6f}")
        expected_action_dim = 8 if args.control_mode == "joint" else 7
        if env.action_space.shape != (expected_action_dim,):
            raise ValueError(f"Expected action space {(expected_action_dim,)}, got {env.action_space.shape}")
        print(f"OK: action_space={env.action_space.shape}")
        processor = IsaacLabTactilePolicyObservationProcessorStep(
            camera_keys="rgb_table,rgb_wrist", control_mode=args.control_mode
        )
        processed = processor._process_observation(observation)
        expected_state_dim = 9 if args.control_mode == "joint" else 11
        assert processed["observation.state"].shape == (1, expected_state_dim)
        assert processed["observation.images.camera1"].shape == (1, 3, 224, 224)
        assert processed["observation.images.camera2"].shape == (1, 3, 224, 224)
        assert processed["observation.gripper_torque"].shape == (1, 30, 1)
        print(
            "OK: processed state/cameras/torque",
            processed["observation.state"].shape,
            processed["observation.images.camera1"].shape,
            processed["observation.images.camera2"].shape,
            processed["observation.gripper_torque"].shape,
        )
    finally:
        env._disconnect()


if __name__ == "__main__":
    main()
