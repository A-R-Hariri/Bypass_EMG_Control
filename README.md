# Bypass EMG Control

Real-time EMG-driven control of a bypass prosthetic hand using Dynamixel servo actuators, a Myo armband, and LibEMG. A deep learning classifier decodes surface EMG signals into hand gestures and translates them into position and velocity commands sent to two Dynamixel motors over a 60 Hz control loop.

---

## Overview

A bypass prosthesis is worn over an intact limb and mechanically couples to it, allowing the user's residual EMG activity to drive an external robotic hand. This system closes the loop between EMG acquisition, real-time gesture classification, and actuator control without any per-session calibration step — the classifier is loaded from a pretrained checkpoint and operates immediately on new users.

The pipeline is:

```
Myo armband (8-ch, 200 Hz)
    -> LibEMG online streamer
    -> OnlineEMGClassifier
    -> UDP socket (probabilities + velocity)
    -> input_thread (gesture -> motor command mapping)
    -> Bypass control loop (60 Hz, Dynamixel SDK sync write)
```

---

## Hardware Requirements

- **Myo Armband** — 8-channel sEMG, 200 Hz, signed 8-bit ADC
- **Dynamixel servos** — two motors, Protocol 1.0
  - `WRIST_ID = 0` — wrist pronation/supination
  - `GRIP_ID = 1` — grip open/close
- **Serial adapter** — connected at `COM6`, 3 Mbaud (update `DEVICENAME` in `bypass.py` for your port)
- **CUDA-capable GPU** — `DEVICE` defaults to `'cuda'` in `utils.py`

---

## Software Dependencies

```
torch
libemg
dynamixel_sdk
numpy
scikit-learn
pandas
matplotlib
joblib
tqdm
```

Install via pip:

```bash
pip install torch libemg dynamixel-sdk numpy scikit-learn pandas matplotlib joblib tqdm
```

---

## Repository Structure

```
Bypass_EMG_Control/
    main.py       -- Entry point: model loading, LibEMG setup, UDP listener, logging
    bypass.py     -- Bypass class: Dynamixel initialization and 60 Hz control loop
    models.py     -- Model definitions and loss functions
    utils.py      -- Hyperparameters, data loaders, training and evaluation routines
```

---

## Models

Three architectures are defined in `models.py`. All scale raw EMG by `/ 128.0` internally to account for the Myo's 8-bit ADC range.

**CNN** (primary model) — multi-horizon dilated convolutional encoder. Three parallel Conv1d branches with dilations 1, 2, and 4 capture features at different temporal scales, concatenated and passed through a final conv layer, adaptive average pooling, and a 128-dimensional embedding head. The classifier head produces 5-class logits.

**CNN_GRL** — CNN with an attached gradient reversal layer (GRL) and a secondary user-identity classifier head. Intended for domain-adversarial training to reduce user-specific features in the embedding space.

**MLP** — simple fully connected baseline operating on handcrafted LibEMG features (default: `WENG`).

Pretrained weights are loaded from the `pickles/` directory at startup. The active model can be switched at runtime by updating `SharedContext.active_model_name`.

---

## Gesture-to-Command Mapping

The classifier outputs one of five classes. `input_thread` in `main.py` maps each predicted gesture to a (wrist, grip) position index:

| Gesture ID | Gesture      | Wrist command | Grip command |
|------------|--------------|---------------|--------------|
| 0          | Rest (NM)    | hold          | hold         |
| 1          | Hand close   | pronation     | hold         |
| 2          | Flexion      | hold          | open (or close if flipped) |
| 3          | Extension    | hold          | close (or open if flipped) |
| 4          | Hand open    | supination    | hold         |

Each DOF has three discrete positions (min, mid, max). Velocity is proportional to the EMG signal strength computed by LibEMG's velocity estimator and is clipped to `[0, 1]` before scaling to the motor's maximum velocity.

The `flip_lr` flag in `SharedContext` swaps the flexion/extension grip direction to accommodate left/right arm differences.

---

## Configuration

Key parameters are defined at the top of their respective files:

**`bypass.py`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEVICENAME` | `'COM6'` | Serial port for Dynamixel USB adapter |
| `BAUDRATE` | `3000000` | Communication baudrate |
| `RATE` | `60` | Control loop frequency (Hz) |
| `WRIST_MIN_POS` / `WRIST_MAX_POS` | `800` / `2500` | Wrist position limits (encoder counts) |
| `GRIP_MIN_POS` / `GRIP_MAX_POS` | `1780` / `2340` | Grip position limits (encoder counts) |
| `WRIST_MAX_TORQUE` / `GRIP_MAX_TORQUE` | `500` / `400` | Torque limits |
| `WRIST_MAX_VEL` / `GRIP_MAX_VEL` | `300` / `200` | Maximum motor velocities |

**`utils.py`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQ` | `40` | EMG window length (samples) |
| `INC` | `2` | Window increment (samples) |
| `CH` | `8` | Number of EMG channels |
| `CLASSES` | `5` | Number of gesture classes |
| `SAMPLING_RATE` | `200` | Myo sampling rate (Hz) |
| `DEVICE` | `'cuda'` | PyTorch device |

---

## Running

1. Connect the Myo armband and ensure the Myo driver/daemon is running.
2. Connect the Dynamixel USB adapter and confirm the correct COM port in `bypass.py`.
3. Place pretrained model weights (`.pt` files) in the `pickles/` directory. The filename must match the model name listed in `model_names` in `main.py`.
4. Run:

```bash
python main.py
```

On startup the system will:
- Load all models listed in `model_names`
- Initialize the LibEMG Myo streamer and online classifier
- Open a UDP socket on port `12346` to receive prediction packets
- Start the input thread for gesture-to-command translation
- Initialize Dynamixel motors (enable torque, set limits)
- Start the 60 Hz control loop
- Log raw EMG to `emg_logs/<NAME>/` and motor state to `bypass_log/<NAME>/`

To stop, trigger an out-of-range position index from the command mapping (or `Ctrl+C`). The control loop will disable motor torque and close the serial port cleanly before exiting.

---

## Data Logging

Two parallel logs are written on every run:

**EMG log** (`emg_logs/<NAME>/bypass_<timestamp>`) — raw streamed EMG via LibEMG's `log_to_file`.

**Bypass log** (`bypass_log/<NAME>/bypass_log_<timestamp>.csv`) — one row per control cycle at 60 Hz containing:

```
time, w_pos_d, w_vel_d, w_trq_d, g_pos_d, g_vel_d, g_trq_d,
w_pos, w_vel, w_trq, g_pos, g_vel, g_trq,
velocity, probs_0, probs_1, probs_2, probs_3, probs_4
```

`_d` columns are commanded (desired) values; unprefixed columns are present (feedback) values read via bulk read from the Dynamixel registers.

---

## Notes

- The `/ 128.0` scaling applied inside every model forward pass is bit-depth normalization for the Myo's signed 8-bit ADC. It is not a learned or per-user normalization step.
- The `MultiModelWrapper` in `main.py` supports hot-swapping between loaded models at runtime by updating `SharedContext.active_model_name` from an external process or UI. The switch is detected on the next forward call.
- A monkey patch is applied to `Protocol1PacketHandler.bulkReadTx` to work around a parameter signature mismatch in some versions of the Dynamixel SDK. If you upgrade the SDK, verify this is still necessary.
- `SharedContext` is a `multiprocessing.Manager().Namespace()` object, allowing state to be shared safely between the main process (Bypass control loop) and the LibEMG subprocess.

---

## License

MIT — see [LICENSE](LICENSE).
