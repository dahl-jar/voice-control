"""
Real-time voice command recognition with keyboard output.
Run: python inference.py
"""

import os
import sys
import time
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast
import numpy as np
import torch
import sounddevice as sd
from collections import deque

from audio_processing import preprocess, get_mel_transform, SAMPLE_RATE, NUM_SAMPLES
from model import VoiceCommandCNN
from config import InferenceConfig


if TYPE_CHECKING:
    from pynput.keyboard import Controller, Key, KeyCode


class CheckpointData(TypedDict):
    """Shape of the model checkpoint used by inference."""

    model_state_dict: Mapping[str, torch.Tensor]
    labels: list[str]
    num_classes: int
    val_acc: float
    epoch: int


def format_keyboard_backend_error(exc: Exception) -> str:
    """Explain common platform-specific pynput failures."""
    base = "Keyboard control is unavailable because pynput could not start."

    if sys.platform.startswith("linux"):
        display = os.environ.get("DISPLAY")
        session_type = os.environ.get("XDG_SESSION_TYPE", "unknown")
        if not display:
            return (
                f"{base} Linux session type is '{session_type}' and DISPLAY is unset. "
                "pynput needs X11/Xwayland, or Linux uinput access as root. "
                f"Original error: {exc}"
            )
        return (
            f"{base} On Linux, pynput depends on X11/Xwayland or uinput access. "
            f"Original error: {exc}"
        )

    return f"{base} Original error: {exc}"


def create_keyboard_controller() -> tuple[
    "Controller", dict[str, "str | Key | KeyCode"]
]:
    """Create the pynput keyboard controller lazily with better diagnostics."""
    try:
        from pynput.keyboard import Controller, Key
    except Exception as exc:
        raise RuntimeError(format_keyboard_backend_error(exc)) from exc

    return Controller(), {
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
        "space": Key.space,
        "escape": Key.esc,
        "Return": Key.enter,
    }


class VoiceController:
    def __init__(self, config: InferenceConfig | None = None, debug: bool = False):
        """
        Loads the model checkpoint, initializes the mel transform (same as training,
        kept on CPU since audio comes from mic), sets up the audio buffer, and builds
        the command index for fast lookup (skipping _unknown and _silence).

        @param config: Inference configuration. Defaults to InferenceConfig().
        @param debug: Whether to print debug output.
        """
        self.config = config or InferenceConfig()
        self.debug = debug
        self.running = False
        self.keyboard, self._key_map = create_keyboard_controller()
        self._last_press_time = 0.0
        self._last_fired = None
        self._vad_threshold = 0.005
        self._prev_prediction = None
        self._streak = 0
        self._quiet_count = 0

        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Device: {self.device}")

        model_path = Path(self.config.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "The trained model is not tracked in git. "
                "Run `python download_model.py` on this machine or copy the checkpoint "
                "into the repo's models directory."
            )

        checkpoint = self._load_checkpoint(model_path)
        self.labels = checkpoint["labels"]
        num_classes = len(self.labels)

        self.model = VoiceCommandCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(
            f"Model loaded (val_acc={checkpoint['val_acc']:.4f}, "
            f"epoch {checkpoint['epoch']})"
        )

        self.mel_transform = get_mel_transform()

        chunk_samples = int(self.config.chunk_duration_sec * SAMPLE_RATE)
        window_samples = int(self.config.window_duration_sec * SAMPLE_RATE)
        self._buffer = deque(maxlen=window_samples)
        self._chunk_samples = chunk_samples
        self._window_samples = window_samples
        self._stride_samples = int(self.config.stride_duration_sec * SAMPLE_RATE)
        self._samples_since_last_classify = 0

        self._command_indices = []
        for i, label in enumerate(self.labels):
            if not label.startswith("_"):
                self._command_indices.append(i)

        print(f"Commands: {[self.labels[i] for i in self._command_indices]}")
        print(f"Key map: {self.config.key_map}")
        print(f"Confidence threshold: {self.config.confidence_threshold}")

    def _press_key(self, key_name: str):
        """Press a keyboard key using pynput (works on macOS, Linux, Windows)."""
        key = self._key_map.get(key_name)
        if key is None:
            key = key_name
        self.keyboard.press(key)
        self.keyboard.release(key)

    def _load_checkpoint(self, model_path: Path) -> CheckpointData:
        """Load the model checkpoint with compatibility for older torch versions."""
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)

        return cast(CheckpointData, checkpoint)

    def _resolve_input_device(self, device: int | None) -> int:
        """Resolve and validate the input device used for calibration and capture."""
        try:
            devices = cast(list[dict[str, object]], list(sd.query_devices()))
        except Exception as exc:
            raise RuntimeError(
                "Unable to query audio devices. Make sure PortAudio is installed and "
                "the microphone is available. On Arch Linux, install the `portaudio` package."
            ) from exc

        input_devices = []
        for index, info in enumerate(devices):
            channel_count = info.get("max_input_channels")
            if isinstance(channel_count, (int, float)) and channel_count > 0:
                input_devices.append(index)

        if not input_devices:
            raise RuntimeError("No input audio devices were found.")

        if device is not None:
            if device not in input_devices:
                raise ValueError(f"Input device {device} is not a valid microphone.")
            return device

        default_device = sd.default.device
        default_input = None
        if isinstance(default_device, (list, tuple)) and default_device:
            default_input = default_device[0]
        elif isinstance(default_device, int):
            default_input = default_device

        if isinstance(default_input, int) and default_input in input_devices:
            return default_input

        return input_devices[0]

    def _dbg(self, msg: str):
        if self.debug:
            sys.stdout.write(f"    DBG {msg}\n")
            sys.stdout.flush()

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice for each audio chunk.

        Performs VAD by checking peak RMS energy in any 200ms segment across the window
        (catches short words like "left" that may not be at the tail). Dispatches
        classification in a separate thread when speech is detected.
        """
        if status:
            print(f"Audio status: {status}")

        samples = indata[:, 0]
        self._buffer.extend(samples.tolist())
        self._samples_since_last_classify += frames

        if (
            self._samples_since_last_classify >= self._stride_samples
            and len(self._buffer) >= self._window_samples
        ):
            self._samples_since_last_classify = 0

            audio = np.array(list(self._buffer), dtype=np.float32)[
                -self._window_samples :
            ]

            seg_len = int(SAMPLE_RATE * 0.2)
            level = 0.0
            for seg_start in range(0, len(audio) - seg_len + 1, seg_len // 2):
                seg = audio[seg_start : seg_start + seg_len]
                seg_rms = np.sqrt(np.mean(seg**2))
                if seg_rms > level:
                    level = seg_rms
            is_speech = level >= self._vad_threshold

            if self.debug:
                bar = "#" * int(min(level / self._vad_threshold, 5) * 10)
                state = "SPEECH" if is_speech else "quiet"
                self._dbg(
                    f"lvl={level:.5f} [{bar:<50}] {state} streak={self._streak} prev={self._prev_prediction} fired={self._last_fired}"
                )

            if not is_speech:
                self._prev_prediction = None
                self._streak = 0
                self._quiet_count += 1
                return
            self._quiet_count = 0

            waveform = torch.from_numpy(audio).unsqueeze(0)
            t_start = time.perf_counter()
            threading.Thread(
                target=self._classify, args=(waveform, t_start), daemon=True
            ).start()

    @torch.no_grad()
    def _classify(self, waveform: torch.Tensor, t_capture: float):
        """
        Run inference on a waveform window.

        Non-command predictions reset the streak. Tracks consecutive same predictions
        regardless of confidence. Fires immediately if very confident (>0.99),
        otherwise requires 2 consecutive agreeing predictions above the threshold.

        @param waveform: Audio tensor from the buffer window.
        @param t_capture: perf_counter timestamp at capture time for latency measurement.
        """
        mel = preprocess(waveform, SAMPLE_RATE, self.mel_transform)
        mel = mel.unsqueeze(0).to(self.device)

        logits = self.model(mel)
        probs = torch.softmax(logits, dim=1)[0]

        best_idx = int(probs.argmax().item())
        best_label = self.labels[best_idx]
        best_prob = probs[best_idx].item()

        self._dbg(f"  classify: {best_label}={best_prob:.3f}")

        if best_label.startswith("_"):
            self._prev_prediction = None
            self._streak = 0
            return

        if best_label == self._prev_prediction:
            self._streak += 1
        else:
            self._prev_prediction = best_label
            self._streak = 1

        needed = 1 if best_prob >= 0.99 else 2
        if self._streak >= needed and best_prob >= self.config.confidence_threshold:
            now = time.time()
            if now - self._last_press_time >= self.config.cooldown_sec:
                key = self.config.key_map.get(best_label)
                if key:
                    latency_ms = (time.perf_counter() - t_capture) * 1000
                    self._last_press_time = now
                    self._streak = 0
                    self._prev_prediction = None
                    self._buffer.clear()
                    self._samples_since_last_classify = 0
                    self._press_key(key)
                    sys.stdout.write(
                        f"  [{best_label}] conf={best_prob:.3f} latency={latency_ms:.1f}ms\n"
                    )
                    sys.stdout.flush()

    def _calibrate_noise(self, device: int | None = None):
        """
        Record 2 seconds of silence to set VAD threshold above background noise.

        Sets the threshold to 2x background RMS, capped between 0.001 and 0.01.
        """
        input_device = self._resolve_input_device(device)
        print("Calibrating background noise — stay QUIET for 2 seconds...")
        time.sleep(0.5)
        audio = sd.rec(
            SAMPLE_RATE * 2,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
        )
        sd.wait()
        bg_rms = np.sqrt(np.mean(audio**2))
        self._vad_threshold = min(bg_rms * 2.0, 0.01)
        self._vad_threshold = max(self._vad_threshold, 0.001)
        print(f"Background RMS: {bg_rms:.6f}, VAD threshold: {self._vad_threshold:.6f}")

    def run(self):
        """Start listening and processing voice commands."""
        input_device = self._resolve_input_device(None)
        self._calibrate_noise(input_device)
        print(f"\nListening on input device {input_device} (rate={SAMPLE_RATE}Hz)...")
        print("Speak a command. Press Ctrl+C to stop.\n")

        self.running = True
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_samples,
                callback=self._audio_callback,
                device=input_device,
            ):
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as e:
            print(f"Error: {e}")
            print(
                "Make sure your microphone is available and sounddevice is installed."
            )

    def test_latency(self):
        """Measure inference latency."""
        print("Measuring inference latency...")
        dummy = torch.randn(1, NUM_SAMPLES)
        mel = preprocess(dummy, SAMPLE_RATE, self.mel_transform)
        mel = mel.unsqueeze(0).to(self.device)

        for _ in range(10):
            self.model(mel)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            self.model(mel)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = sum(times) / len(times)
        print(
            f"Inference latency: {avg:.2f}ms avg "
            f"(min={min(times):.2f}ms, max={max(times):.2f}ms)"
        )
        print(
            f"Stride: {self.config.stride_duration_sec * 1000:.0f}ms "
            f"(classify every {self.config.stride_duration_sec * 1000:.0f}ms)"
        )
        print(
            f"Total expected latency: ~{avg + self.config.stride_duration_sec * 1000:.0f}ms"
        )


if __name__ == "__main__":
    config = InferenceConfig()
    debug = "--debug" in sys.argv
    controller = VoiceController(config, debug=debug)

    if "--latency" in sys.argv:
        controller.test_latency()
    else:
        controller.run()
