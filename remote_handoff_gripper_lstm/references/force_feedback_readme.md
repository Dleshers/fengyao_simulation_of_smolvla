- ### Part 1: Code Modifications

  1. **Add LSTM Architecture**: Defined `ForceLSTMEncoder` at the beginning of the file, which is a network designed to convert time-series signals into a 16-dimensional latent vector.
  2. **Load and Freeze Weights**: During model initialization, it automatically loads the `torque_16d_encoder.pt` weights. Simultaneously, it **locks these parameters (disables gradient updates)** so that it functions strictly as a stable feature extractor. (Relative path for the stored .pt file: `"trained_lstm_weights/torque_16d_encoder.pt"`)
  3. **Add Alignment Interface**: Added a `force_proj` linear layer to project the 16-dimensional tactile features, aligning them with the token lengths of vision and language modalities so they can be processed by the cross-attention mechanism.
  4. **Establish Raw Data Pipeline**: Modified the forward pass (training) and action sampling (inference) functions. The model can now directly ingest raw torque sequences named `observation.gripper_torque`. It instantaneously converts this into 16-dimensional features internally before passing them to the Action Expert for decision-making.

  ### Part 2: Data Flow and Practical Configuration

  #### 1. Real-Robot Data Collection: Direct Recording of Time-Series Windows (Core Modification)

  In the low-level physical data collection and recording scripts, the "single-frame, single-point" float recording method has been deprecated. Instead, it is replaced by **maintaining a real-time time window in memory**:

  - **Queue Maintenance**: A torque/current data queue with a maximum length of 30 is maintained in the low-level code (e.g., Python/ROS node), continuously pushing the latest float values into the queue at the hardware's maximum frequency (e.g., 500Hz).
  - **Synchronization and Alignment**: At the exact moment the camera triggers an exposure to save an image (e.g., at 30Hz), the system directly captures the 30 historical data points currently in the queue and concatenates them into a `[30, 1]` matrix.
  - **Cold Start Handling (Padding)**: At the very beginning of the recording, when there are fewer than 30 frames of historical data, **a "duplicate first frame" logic must be implemented in the collection script** (i.e., padding the gap with the initial torque value). Padding with zeros is strictly prohibited to prevent the model from misinterpreting it as a rigid collision.
  - **Final Storage**: When saved to disk, the torque data corresponding to each image frame is directly formatted as a perfect `[30, 1]` matrix.

  #### 2. Data Packaging and Metadata Validation

  Package the recorded MP4 videos, joint states, and log files containing the `[30, 1]` matrices into the Parquet dataset format supported by Hugging Face.

  - **Core Naming Convention**: The key name for the torque features **must** be strictly set to `observation.gripper_torque`.
  - **Low-Level Validation**: After packaging, be sure to inspect the `meta/info.json` file in the dataset's root directory. Verify that the `shape` attribute of the `observation.gripper_torque` field is correctly recorded as `[30, 1]`. If the recording framework defaults to recognizing it as another shape, manually correct it to `[30, 1]` to prevent reading errors from the Arrow engine.

  #### 3. Launching Training: End-to-End Feature Learning

  Because the data has been perfectly windowed at the source, **no offline secondary slicing processing is required**. You can directly run `lerobot-train`.

  - During training, the DataLoader will directly feed the ready-made `[30, 1]` matrices into the model.
  - The internally frozen LSTM instantly converts them into 16-dimensional features, and the model autonomously begins to learn "how the gripper and robotic arm should adjust their poses when the gripper experiences specific time-series fluctuations due to external forces."

  #### 4. Real-Robot Deployment: Seamless Inference

  When deploying the fine-tuned model to the real-robot industrial PC:

  - The low-level hardware maintains a double-ended queue (`deque(maxlen=30)`) via a Python/ROS node, continuously storing real-time gripper current values at a high frequency, such as 500Hz.
  - When 30Hz inference is triggered, the data currently in the queue is directly converted into a `[1, 30, 1]` tensor, packaged into `batch["observation.gripper_torque"]`, and passed into `policy.select_action(batch)`.
  - Upon receiving the data, the model completes the end-to-end physical control loop in milliseconds: "**Read time-series current -> Convert to 16D feature -> Fuse with visual instructions -> Output real physical actions**".