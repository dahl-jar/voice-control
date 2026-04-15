"""
Real-time voice command recognition with keyboard output.
Run: python inference.py
"""

import logging
import sys
import time
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast
import numpy as np
import torch
import sounddevice as sd
from collections import deque

from audio_processing import preprocess, get_mel_transform, SAMPLE_RATE, NUM_SAMPLES
from diagnostics import diagnose, report_model_footprint
from keyboard_backend import load_keyboard_backend
from log_config import configure_logging
from model import VoiceCommandCNN
from config import InferenceConfig


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pynput.keyboard import Controller, Key, KeyCode


class CheckpointData(TypedDict):
    """Shape of the model checkpoint used by inference."""

    model_state_dict: Mapping[str, torch.Tensor]
    labels: list[str]
    val_acc: float
    epoch: int
    num_classes: NotRequired[int]


def create_keyboard_controller() -> tuple[
    "Controller", dict[str, "str | Key | KeyCode"]
]:
    """Create the pynput keyboard controller lazily with better diagnostics."""
    controller, key = load_keyboard_backend()

    return controller, {
        "up": key.up,
        "down": key.down,
        "left": key.left,
        "right": key.right,
        "space": key.space,
        "escape": key.esc,
        "Return": key.enter,
    }


def get_default_input_device() -> int | None:
    """Return the default input device index from sounddevice, if available."""
    default_device = sd.default.device
    if isinstance(default_device, int):
        return default_device if default_device >= 0 else None

    try:
        default_input = int(default_device[0])
    except (IndexError, KeyError, TypeError, ValueError):
        return None

    return default_input if default_input >= 0 else None


class VoiceController:
    def __init__(self, config: InferenceConfig | None = None, debug: bool = False):
        """
        Wire up everything the hot path needs before the mic opens.
        Mel transform stays on CPU — audio arrives from sounddevice
        in numpy land and round-tripping to GPU isn't worth it.

        @param config: Inference configuration. Defaults to InferenceConfig().
        @param debug: Print per-frame VAD/prediction traces for tuning.
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
        self._capture_sample_rate = SAMPLE_RATE
        self._input_device = None

        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device: {self.device}")

        model_path = Path(self.config.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "The trained model is not tracked in git. "
                "Run `python download_model.py` on this machine or copy the checkpoint "
                "into the repo's models directory."
            )

        with diagnose("model_load") as load_diag:
            checkpoint = self._load_checkpoint(model_path)
            self.labels = checkpoint["labels"]
            num_classes = len(self.labels)

            self.model = VoiceCommandCNN(num_classes=num_classes).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        logger.info(
            f"Model loaded (val_acc={checkpoint['val_acc']:.4f}, "
            f"epoch {checkpoint['epoch']})"
        )

        logger.info(load_diag.format_line())
        report_model_footprint(self.model)

        self.mel_transform = get_mel_transform()

        self._configure_capture_timing(SAMPLE_RATE)

        self._command_indices = []
        for i, label in enumerate(self.labels):
            if not label.startswith("_"):
                self._command_indices.append(i)

        logger.info(f"Commands: {[self.labels[i] for i in self._command_indices]}")
        logger.info(f"Key map: {self.config.key_map}")
        logger.info(f"Confidence threshold: {self.config.confidence_threshold}")

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

        default_input = get_default_input_device()
        if isinstance(default_input, int) and default_input in input_devices:
            return default_input

        return input_devices[0]

    def _configure_capture_timing(self, sample_rate: int) -> None:
        """Rebuild streaming buffers for the active microphone sample rate."""
        self._capture_sample_rate = sample_rate
        self._chunk_samples = int(self.config.chunk_duration_sec * sample_rate)
        self._window_samples = int(self.config.window_duration_sec * sample_rate)
        self._stride_samples = int(self.config.stride_duration_sec * sample_rate)
        self._buffer = deque(maxlen=self._window_samples)
        self._samples_since_last_classify = 0

    def _prepare_input_stream(self, device: int | None) -> int:
        """Resolve the selected microphone and capture at its native input rate."""
        input_device = self._resolve_input_device(device)
        device_info = cast(dict[str, object], sd.query_devices(input_device))
        default_sample_rate = device_info.get("default_samplerate")
        if (
            not isinstance(default_sample_rate, (int, float))
            or default_sample_rate <= 0
        ):
            raise RuntimeError(
                f"Input device {input_device} does not report a valid sample rate."
            )

        capture_sample_rate = int(round(default_sample_rate))
        sd.check_input_settings(
            device=input_device,
            samplerate=capture_sample_rate,
            channels=1,
            dtype="float32",
        )

        self._input_device = input_device
        self._configure_capture_timing(capture_sample_rate)
        return input_device

    def _print_debug(self, msg: str):
        logger.debug(msg)

    def _audio_callback(self, indata, frames, time_info, status):
        """
        sounddevice callback. Ring-buffers audio, runs VAD every stride.

        VAD takes peak RMS across 200ms sub-segments (not full-window)
        so short words like "left" don't get diluted by surrounding
        silence. Inference runs on a daemon thread — blocking this
        callback drops audio frames.
        """
        if status:
            logger.warning(f"Audio status: {status}")

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

            seg_len = int(self._capture_sample_rate * 0.2)
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
                self._print_debug(
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
        Forward pass + debouncing. Runs off the audio thread.

        Debounce: conf >= 0.99 fires on a single hit, otherwise require
        two consecutive agreeing predictions. `_unknown`/`_silence`
        resets the streak. Buffer is cleared after firing to prevent
        double-triggers on the same window.

        @param waveform: Audio tensor from the sliding window.
        @param t_capture: perf_counter stamp at capture, used for end-to-end latency.
        """
        mel = preprocess(waveform, self._capture_sample_rate, self.mel_transform)
        mel = mel.unsqueeze(0).to(self.device)

        logits = self.model(mel)
        probs = torch.softmax(logits, dim=1)[0]

        best_idx = int(probs.argmax().item())
        best_label = self.labels[best_idx]
        best_prob = probs[best_idx].item()

        self._print_debug(f"  classify: {best_label}={best_prob:.3f}")

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
                    logger.info(
                        f"[{best_label}] conf={best_prob:.3f} latency={latency_ms:.1f}ms"
                    )

    def _calibrate_noise(self, device: int | None = None):
        """
        Sample 2s of the room, set VAD threshold to 2x bg RMS,
        clamped to [0.001, 0.01]. Clamp range covers bad mics that
        report near-zero or absurdly high floors.
        """
        input_device = self._prepare_input_stream(device)
        logger.info("Calibrating background noise — stay QUIET for 2 seconds...")
        time.sleep(0.5)
        audio = sd.rec(
            self._capture_sample_rate * 2,
            samplerate=self._capture_sample_rate,
            channels=1,
            dtype="float32",
            device=input_device,
        )
        sd.wait()
        bg_rms = float(np.sqrt(np.mean(audio**2)))
        raw_threshold = bg_rms * 2.0
        vad_ceiling = 0.01
        vad_floor = 0.001
        self._vad_threshold = max(min(raw_threshold, vad_ceiling), vad_floor)
        logger.info(f"Background RMS: {bg_rms:.6f}, VAD threshold: {self._vad_threshold:.6f}")

        if raw_threshold > vad_ceiling:
            logger.warning(
                f"Background noise is high (bg_rms={bg_rms:.6f}); VAD threshold clamped "
                f"to ceiling {vad_ceiling}. Quiet commands may be missed — consider "
                "moving to a quieter environment or using a closer/better mic."
            )
        elif raw_threshold < vad_floor:
            logger.warning(
                f"Background noise is very low (bg_rms={bg_rms:.6f}); VAD threshold "
                f"clamped to floor {vad_floor}. This can cause false triggers on tiny "
                "mic spikes — check that calibration ran without you speaking."
            )

    def run(self):
        """Open the mic and block until Ctrl+C. Sleep loop keeps
        InputStream's context alive — returning early closes the stream."""
        input_device = self._prepare_input_stream(None)
        self._calibrate_noise(input_device)
        logger.info(
            f"Listening on input device {input_device} "
            f"(rate={self._capture_sample_rate}Hz, model={SAMPLE_RATE}Hz)"
        )
        logger.info("Speak a command. Press Ctrl+C to stop.")

        self.running = True
        try:
            with sd.InputStream(
                samplerate=self._capture_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_samples,
                callback=self._audio_callback,
                device=input_device,
            ):
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Stopped.")

    def test_latency(self):
        """Measure inference latency."""
        logger.info("Measuring inference latency...")
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
        logger.info(
            f"Inference latency: {avg:.2f}ms avg "
            f"(min={min(times):.2f}ms, max={max(times):.2f}ms)"
        )
        logger.info(
            f"Stride: {self.config.stride_duration_sec * 1000:.0f}ms "
            f"(classify every {self.config.stride_duration_sec * 1000:.0f}ms)"
        )
        logger.info(
            f"Total expected latency: ~{avg + self.config.stride_duration_sec * 1000:.0f}ms"
        )


if __name__ == "__main__":
    configure_logging()
    config = InferenceConfig()
    debug = "--debug" in sys.argv
    if debug:
        logger.setLevel(logging.DEBUG)
    controller = VoiceController(config, debug=debug)

    if "--latency" in sys.argv:
        controller.test_latency()
    else:
        controller.run()
