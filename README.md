# Bypass EMG Control

Real-time myoelectric control of a bypass prosthetic hand. Surface EMG from a Myo armband is classified by a pretrained PyTorch model through LibEMG, and the resulting class probabilities and contraction intensity are converted into position and velocity commands for two Dynamixel servos running in a 60 Hz closed-loop.

A bypass prosthesis is worn over an intact limb, so an unimpaired subject can drive an external hand with their own EMG. No per-session calibration or screen-guided training is required in the default configuration: the classifier is loaded from a checkpoint and runs immediately on a new user.

## Signal path

```
Myo armband (8 channels, 200 Hz, signed 8-bit)
  -> libemg.streamers.myo_streamer
  -> OnlineDataHandler
  -> OnlineEMGClassifier (window 40, increment 2, probability output)
  -> UDP 127.0.0.1:12346
  -> input_thread (probabilities + velocity -> position index and speed)
  -> SharedContext (multiprocessing Namespace)
  -> Bypass.run (60 Hz Dynamixel sync write and bulk read)
```

The classifier runs inside a LibEMG subprocess. The control loop runs in the main process. State crosses the boundary through a `multiprocessing.Manager().Namespace()` object, which is also readable by an external Fitts law test process.

## Contents

```
main.py     Entry point. Model loading, multi-model wrapper, LibEMG setup,
            UDP listener thread, gesture to command mapping, EMG logging.
bypass.py   Bypass class. Dynamixel setup, control table constants,
            60 Hz control loop, motor state logging, shutdown.
models.py   Six architectures and six loss functions.
utils.py    Global constants, hyperparameters, data loaders, training and
            evaluation routines, Fitts test parameter dictionary.
```

The `pickles/`, `emg_logs/` and `bypass_log/` directories are gitignored.

## Hardware

| Item | Detail |
|------|--------|
| EMG | Myo armband, 8 channels, 200 Hz, signed 8-bit ADC |
| Actuators | Two Dynamixel servos, Protocol 1.0 |
| Wrist motor | `WRIST_ID = 0`, pronation and supination |
| Grip motor | `GRIP_ID = 1`, open and close |
| Interface | USB serial adapter on `COM6` at 3 Mbaud |
| Compute | CUDA device, `DEVICE = 'cuda'` in `utils.py` |

`main.py` pins the process to GPU 0 through `CUDA_VISIBLE_DEVICES`.

## Dependencies

```
torch
libemg
dynamixel_sdk
numpy
scikit-learn
pandas
matplotlib
tqdm
joblib
```

```bash
pip install torch libemg dynamixel-sdk numpy scikit-learn pandas matplotlib tqdm joblib
```

## Configuration

Constants in `bypass.py`:

| Name | Default | Meaning |
|------|---------|---------|
| `DEVICENAME` | `'COM6'` | Serial port of the USB adapter |
| `BAUDRATE` | `3000000` | Bus baudrate |
| `RATE` | `60` | Control loop frequency in Hz |
| `NAME` | `'aibme'` | Subject subfolder for motor logs |
| `WRIST_MIN_POS`, `WRIST_MAX_POS` | `800`, `2500` | Wrist travel in encoder counts |
| `GRIP_MIN_POS`, `GRIP_MAX_POS` | `1780`, `2340` | Grip travel in encoder counts |
| `WRIST_MAX_TORQUE`, `GRIP_MAX_TORQUE` | `500`, `400` | Torque limit registers |
| `WRIST_MAX_VEL`, `GRIP_MAX_VEL` | `300`, `200` | Velocity scaling ceiling |

`WRIST_POS` and `GRIP_POS` are built as three-element lists holding the minimum, the midpoint and the maximum. Position commands index into these lists.

Constants in `utils.py`:

| Name | Default | Meaning |
|------|---------|---------|
| `SEQ` | `40` | Window length in samples |
| `INC` | `2` | Window increment in samples |
| `CH` | `8` | EMG channels |
| `CLASSES` | `5` | Gesture classes |
| `SAMPLING_RATE` | `200` | Myo sampling rate in Hz |
| `NAME` | `'1'` | Subject subfolder for EMG logs, per-user checkpoints and Fitts logs |
| `PICKLE_PATH` | `'pickles'` | Checkpoint and threshold directory |
| `SGT_PATH` | `user_sgt/<NAME>` | Per-user checkpoints for screen guided training models |
| `DATA_PATH` | `emg_logs/<NAME>` | Raw EMG log directory |
| `FEATURE_LIST` | `['WENG']` | Feature set for the feature-based model |
| `EPOCHS`, `BATCH_SIZE` | `100`, `512` | Training budget |
| `DROPOUT`, `PATIENCE` | `0.2`, `5` | Regularization and early stopping |
| `LR_INIT`, `LR_MIN`, `LR_FACTOR`, `LR_PATIENCE` | `1e-4`, `1e-5`, `0.6`, `4` | Adam and plateau scheduler settings |

`NAME` is defined in both files. Set both to the same subject identifier to keep EMG and motor logs under one folder name.

`PARAMS` in `utils.py` holds the Fitts law test configuration: frame rate, target radii and distances, ring radii, hold and timeout frames, screen size, an optional physics block with mass, damping and acceleration limits, and a velocity constant. It is copied into `SharedContext.params` at startup so a separate test process can read it.

## Models

All models divide the input by 128.0 in the forward pass. This is bit-depth normalization for the Myo's signed 8-bit range, not a learned or per-user scaling step.

`CNN` is the model used for online control. Three parallel `Conv1d` branches over the raw 8-channel window with kernel size 8 and dilations 1, 2 and 4 give three receptive fields over the same 40-sample input. The branches are concatenated to 96 channels, passed through a fourth convolution to 128 channels, reduced by adaptive average pooling, and projected through a GELU-activated fully connected layer to a 128-dimensional embedding. A single linear layer produces the 5 class logits. `forward` optionally returns the embedding, the logits, or both, which is the form the embedding-based losses expect.

`CNN_GRL` is the same encoder with a gradient reversal layer and a second linear head over 306 subject identities. Reversing the gradient on that head during training penalizes subject-identifiable structure in the embedding. Set `num_grl` to the number of training subjects.

`MLP` is the feature-based model. Four fully connected layers with ReLU and dropout down to a 64-dimensional embedding. `load_all_models` constructs it with 48 input features.

`CNNBasic`, `LSTM` and `TransformerEMG` are additional architectures in `models.py` for offline comparison.

Checkpoint loading in `load_all_models` is driven by substring matches on the checkpoint name:

| Name contains | Architecture | Loaded from |
|---------------|--------------|-------------|
| `grl` | `CNN_GRL` | `pickles/` |
| `mlp` | `MLP(48)` | `user_sgt/<NAME>/` if the name also contains `within`, otherwise `pickles/` |
| `within` | `CNN` | `user_sgt/<NAME>/` |
| anything else | `CNN` | `pickles/` |

`model_names` in `main.py` lists which checkpoints to load, currently `cnn_raw`. Every listed name must exist as `<name>.pt` in the corresponding directory.

## Losses

Defined in `models.py`. The first three operate on logits and targets and pass directly as `loss_fn` to `train`. The last three take embeddings, so call `train` with `return_emb=True` and `return_logits=True`.

| Class | Signature | Behaviour |
|-------|-----------|-----------|
| `RestLoss` | `(logits, targets)` | Cross entropy with two multipliers: `alpha1` on any wrong non-rest prediction, `alpha2` on rest samples predicted as active. Targets false activations. |
| `EqLoss` | `(logits, targets)` | Mean cross entropy plus `alpha` times the variance of per-class mean loss normalized by its mean. Penalizes uneven class difficulty. |
| `CVaRLoss` | `(logits, targets)` | Mean over the worst `alpha` fraction of per-sample losses. Optimizes the tail rather than the average. |
| `TripletLoss` | `(z, labels, subjects)` | Batch-hard triplets on cosine distance. Positives are the same gesture from a different subject, negatives are a different gesture from the same subject, which pushes the embedding toward gesture structure and away from subject structure. `w_soft` adds the conventional same-subject positive, different-subject negative term. |
| `PrototypeLoss` | `(emb, logits, labels)` | Cross entropy blended with mean squared distance to the per-class embedding centroid. |
| `OneVsAllLoss` | `(emb, logits, labels)` | Cross entropy blended with a softmax over negative squared distances to all class centroids, pulling toward the correct centroid and pushing away from the rest. |

## Training utilities

`utils.py` provides `create_loader`, `train` and `evaluate`, called from a training script or notebook.

`train` uses Adam, mixed precision through `autocast` and `GradScaler`, `ReduceLROnPlateau` on validation loss, early stopping on `PATIENCE` epochs without improvement, and restores the best state dict before returning. Setting `return_emb` and `return_logits` switches the forward call and the loss call to the three-argument form used by the embedding losses.

`remap_labels` applies the fixed mapping `{0:1, 1:4, 2:0, 3:3, 4:2}` to align dataset label order with the class order the control mapping expects.

## Online classifier and velocity

`main.py` wraps the loaded models in `MultiModelWrapper`, an `nn.Module` exposing `forward`, `predict` and `predict_proba`. On every forward call it reads `SharedContext.active_model_name` and swaps the active model if the name changed, so models can be switched at runtime from another process without restarting the streamer.

The wrapper is passed to `libemg.emg_predictor.EMGClassifier`. Velocity output is enabled through `add_velocity`, and the per-class thresholds are then loaded from `pickles/th_max_dic.npy` and `pickles/th_min_dic.npy` into `th_max_dic` and `th_min_dic`, which lets a pretrained model produce proportional speed without a calibration recording.

`OnlineEMGClassifier` is created with `features=None`, so the raw window is passed to the model, and `output_format='probabilities'`. Each decision is sent as a space-separated UDP datagram to `127.0.0.1:12346`:

```
p0 p1 p2 p3 p4 velocity timestamp
```

`input_thread` reads the socket with `select`, routes packets by model category (`normal`, `within_cnn`, `within_mlp`) so the active model's stream is the one acted on, takes `argmax` over the five probabilities, and clips the velocity field to `[0, 1]`.

## Gesture to command mapping

| Class | Gesture | Wrist | Grip |
|-------|---------|-------|------|
| 0 | No motion | hold | hold |
| 1 | Hand close | hold | index 0 |
| 2 | Wrist flexion | index 0, or index 2 if `flip_lr` | hold |
| 3 | Wrist extension | index 2, or index 0 if `flip_lr` | hold |
| 4 | Hand open | hold | index 2 |

Indices refer to `WRIST_POS` and `GRIP_POS`. Index 0 is the minimum encoder count and index 2 is the maximum. `flip_lr` in `SharedContext` swaps the wrist direction for left versus right arm mounting.

A holding degree of freedom keeps its previous position index and gets speed 0. The moving degree of freedom gets the clipped classifier velocity, which the control loop scales by `WRIST_MAX_VEL` or `GRIP_MAX_VEL`.

## Control loop

`Bypass.setup` opens the port, sets the baudrate, enables torque on both motors and writes the velocity and torque limit registers. `Bypass.run` then holds a 60 Hz cycle using `perf_counter` with a fixed interval and sleeps the remainder of each period.

Each cycle:

1. Reads `wrist_pos`, `grip_pos`, `speedW` and `speedG` from `SharedContext`.
2. Scales the speeds to goal velocities. A goal velocity of 0 means maximum speed on Protocol 1.0, so a computed 0 is raised to 1.
3. Sync writes both goal positions, then sync writes both goal velocities.
4. Bulk reads 6 bytes from `ADDR_PRESENT_POSITION` on each motor, covering present position, velocity and torque in one transaction.
5. Writes one CSV row.

The loop exits when either position index falls outside `[0, 1, 2]`, which is how an external process requests a clean stop. On exit it closes the log, disables torque on both motors and closes the port.

## Logging

Two logs are written per run.

Raw EMG through LibEMG's `log_to_file`:

```
emg_logs/<NAME>/bypass_<YYYY-MM-DD_HH-MM-SS>
```

Motor and decision state, one row per control cycle:

```
bypass_log/<NAME>/bypass_log_<YYYY-MM-DD_HH-MM-SS>.csv
```

```
time, w_pos_d, w_vel_d, w_trq_d, g_pos_d, g_vel_d, g_trq_d,
w_pos, w_vel, w_trq, g_pos, g_vel, g_trq,
velocity, probs_0, probs_1, probs_2, probs_3, probs_4
```

Columns ending in `_d` are commanded values. The unsuffixed columns are the feedback values from the bulk read. `w_pos_d` and `g_pos_d` are position indices, `w_pos` and `g_pos` are raw encoder counts.

## Running

1. Connect the Myo and confirm its driver or daemon is running.
2. Connect the Dynamixel adapter and set `DEVICENAME` to the correct port.
3. Place the checkpoints named in `model_names` into `pickles/` as `<name>.pt`, together with `th_max_dic.npy` and `th_min_dic.npy`.
4. Set `NAME` in `bypass.py` and `utils.py`.

```bash
python main.py
```

Startup order is: load checkpoints, build the wrapper and classifier, load velocity thresholds, start the Myo streamer, start the online classifier, bind the UDP socket, start the input thread, open the EMG log, initialize the motors, enter the control loop.

Stop by driving a position index out of range from the controlling process, or with `Ctrl+C`.

## Implementation notes

`Protocol1PacketHandler.bulkReadTx` is patched at import time in `bypass.py` to absorb extra positional arguments, which keeps the bulk read compatible across Dynamixel SDK versions.

Fields on `SharedContext` written by `main.py`:

| Field | Consumer |
|-------|----------|
| `speedW`, `speedG`, `wrist_pos`, `grip_pos` | control loop |
| `probs`, `velocity` | control loop, for logging |
| `flip_lr` | `input_thread` |
| `active_model_name` | `MultiModelWrapper` |
| `available_models`, `params`, `speed_multiplier` | external test process |

Seeds are fixed to 13 in `main.py` for `random`, `numpy` and `torch`.

## License

MIT. See [LICENSE](LICENSE).