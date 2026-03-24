"""
Real-time voice command recognition with keyboard output.
Run: python inference.py
"""

import sys
import time
import threading
import numpy as np
import torch
import sounddevice as sd
from collections import deque
from pynput.keyboard import Controller, Key

from audio_processing import preprocess, get_mel_transform, SAMPLE_RATE, NUM_SAMPLES
from model import VoiceCommandCNN
from config import InferenceConfig

PYNPUT_KEY_MAP = {
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
        self.keyboard = Controller()
        self._last_press_time = 0.0
        self._last_fired = None
        self._vad_threshold = 0.005
        self._prev_prediction = None
        self._streak = 0
        self._quiet_count = 0

        self.device = torch.device(self.config.device
                                    if torch.cuda.is_available()
                                    else "cpu")
        print(f"Device: {self.device}")

        checkpoint = torch.load(self.config.model_path, map_location=self.device,
                                weights_only=True)
        self.labels = checkpoint["labels"]
        num_classes = checkpoint["num_classes"]

        self.model = VoiceCommandCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Model loaded (val_acc={checkpoint['val_acc']:.4f}, "
              f"epoch {checkpoint['epoch']})")

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
        key = PYNPUT_KEY_MAP.get(key_name)
        if key is None:
            key = key_name
        self.keyboard.press(key)
        self.keyboard.release(key)

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

        if (self._samples_since_last_classify >= self._stride_samples
                and len(self._buffer) >= self._window_samples):
            self._samples_since_last_classify = 0

            audio = np.array(list(self._buffer), dtype=np.float32)[-self._window_samples:]

            seg_len = int(SAMPLE_RATE * 0.2)
            level = 0.0
            for seg_start in range(0, len(audio) - seg_len + 1, seg_len // 2):
                seg = audio[seg_start:seg_start + seg_len]
                seg_rms = np.sqrt(np.mean(seg ** 2))
                if seg_rms > level:
                    level = seg_rms
            is_speech = level >= self._vad_threshold

            if self.debug:
                bar = "#" * int(min(level / self._vad_threshold, 5) * 10)
                state = "SPEECH" if is_speech else "quiet"
                self._dbg(f"lvl={level:.5f} [{bar:<50}] {state} streak={self._streak} prev={self._prev_prediction} fired={self._last_fired}")

            if not is_speech:
                self._prev_prediction = None
                self._streak = 0
                self._quiet_count += 1
                return
            self._quiet_count = 0

            waveform = torch.from_numpy(audio).unsqueeze(0)
            t_start = time.perf_counter()
            threading.Thread(target=self._classify, args=(waveform, t_start),
                           daemon=True).start()

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

        best_idx = probs.argmax().item()
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
        if (self._streak >= needed
                and best_prob >= self.config.confidence_threshold):
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

    def _calibrate_noise(self):
        """
        Record 2 seconds of silence to set VAD threshold above background noise.

        Sets the threshold to 2x background RMS, capped between 0.001 and 0.01.
        """
        print("Calibrating background noise — stay QUIET for 2 seconds...")
        time.sleep(0.5)
        audio = sd.rec(SAMPLE_RATE * 2, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        bg_rms = np.sqrt(np.mean(audio ** 2))
        self._vad_threshold = min(bg_rms * 2.0, 0.01)
        self._vad_threshold = max(self._vad_threshold, 0.001)
        print(f"Background RMS: {bg_rms:.6f}, VAD threshold: {self._vad_threshold:.6f}")

    def run(self):
        """Start listening and processing voice commands."""
        self._calibrate_noise()
        print(f"\nListening on default mic (rate={SAMPLE_RATE}Hz)...")
        print("Speak a command. Press Ctrl+C to stop.\n")

        self.running = True
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_samples,
                callback=self._audio_callback,
            ):
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure your microphone is available and sounddevice is installed.")

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
        print(f"Inference latency: {avg:.2f}ms avg "
              f"(min={min(times):.2f}ms, max={max(times):.2f}ms)")
        print(f"Stride: {self.config.stride_duration_sec*1000:.0f}ms "
              f"(classify every {self.config.stride_duration_sec*1000:.0f}ms)")
        print(f"Total expected latency: ~{avg + self.config.stride_duration_sec*1000:.0f}ms")


if __name__ == "__main__":
    config = InferenceConfig()
    debug = "--debug" in sys.argv
    controller = VoiceController(config, debug=debug)

    if "--latency" in sys.argv:
        controller.test_latency()
    else:
        controller.run()
