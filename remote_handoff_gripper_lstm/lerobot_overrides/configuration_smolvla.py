# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Optional narrow setup for gripper-only tactile experiments.
    gripper_only_tactile: bool = False
    tactile_torque_key: str = "observation.gripper_torque"
    wrist_camera_key: str = f"{OBS_IMAGES}.wrist"

    # Torque-window ablation: raw history -> LSTM -> one action-expert suffix token.
    # Keep this False for the visual-only baseline.
    use_torque_lstm: bool = False
    torque_window_key: str = "observation.gripper_torque"
    torque_window_size: int = 30
    torque_input_dim: int = 1
    torque_lstm_hidden_dim: int = 64
    torque_lstm_output_dim: int = 16
    torque_lstm_num_layers: int = 2
    train_torque_lstm: bool = True
    torque_lstm_weights_path: str | None = None

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.gripper_only_tactile and self.input_features and self.output_features:
            self._validate_gripper_only_tactile_setup()
        if self.use_torque_lstm:
            if self.torque_window_size <= 0 or self.torque_input_dim <= 0:
                raise ValueError("Torque window size and input dimension must be positive.")
            if self.torque_lstm_hidden_dim <= 0 or self.torque_lstm_output_dim <= 0:
                raise ValueError("Torque LSTM hidden and output dimensions must be positive.")
            if self.torque_lstm_num_layers <= 0:
                raise ValueError("Torque LSTM must have at least one layer.")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

        if self.gripper_only_tactile:
            self._validate_gripper_only_tactile_setup()
        if self.use_torque_lstm:
            torque_feature = self.input_features.get(self.torque_window_key)
            expected_shape = (self.torque_window_size, self.torque_input_dim)
            if torque_feature is None:
                raise ValueError(
                    f"`use_torque_lstm=True` requires input feature '{self.torque_window_key}'."
                )
            if torque_feature.shape != expected_shape:
                raise ValueError(
                    f"Torque feature '{self.torque_window_key}' must have shape {expected_shape}, "
                    f"got {torque_feature.shape}."
                )

    def _validate_gripper_only_tactile_setup(self) -> None:
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if len(image_features) != 1 or image_features[0] != self.wrist_camera_key:
            raise ValueError(
                "`gripper_only_tactile=True` expects exactly one wrist camera feature "
                f"named '{self.wrist_camera_key}', got {image_features}."
            )

        action_feature = self.output_features.get("action")
        if action_feature is None or action_feature.shape != (1,):
            raise ValueError(
                "`gripper_only_tactile=True` expects output feature 'action' with shape (1,), "
                f"got {action_feature}."
            )

        if self.tactile_torque_key not in self.input_features:
            raise ValueError(
                "`gripper_only_tactile=True` expects raw torque input feature "
                f"'{self.tactile_torque_key}'."
            )

        if self.empty_cameras != 0:
            raise ValueError("`gripper_only_tactile=True` expects `empty_cameras=0`.")

        if self.adapt_to_pi_aloha:
            raise ValueError("`gripper_only_tactile=True` expects `adapt_to_pi_aloha=False`.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
