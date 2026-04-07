"""
Voice Command GUI — thin tkinter wrapper around inference.py's VoiceController.
Run: python ui.py
"""

import threading
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import sounddevice as sd

from inference import VoiceController, format_keyboard_backend_error
from config import InferenceConfig

if TYPE_CHECKING:
    from pynput.keyboard import Controller, Key, KeyCode


SPECIAL_KEYS = {}


def create_ui_keyboard_controller() -> tuple[
    "Controller", dict[str, "str | Key | KeyCode"]
]:
    """Create the GUI keyboard backend lazily with platform-aware errors."""
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
        "esc": Key.esc,
        "enter": Key.enter,
        "tab": Key.tab,
        "shift": Key.shift,
        "ctrl": Key.ctrl,
        "alt": Key.alt,
    }


def parse_key(key_str):
    """
    @param key_str: Key name like "up", "space", or a single character.
    @returns: pynput Key constant, single char, or None if invalid.
    """
    key_str = key_str.strip().lower()
    if key_str in SPECIAL_KEYS:
        return SPECIAL_KEYS[key_str]
    if len(key_str) == 1:
        return key_str
    return None


class VoiceCommandApp:
    """
    @param master: Root tk.Tk instance.
    """

    def __init__(self, master):
        self.root = master
        self.root.title("Voice Command Controller")
        self.root.resizable(False, False)

        style = ttk.Style()
        for theme in ("aqua", "clam", "vista", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break

        self.config = InferenceConfig()
        self.commands = {}
        self.keyboard, special_keys = create_ui_keyboard_controller()
        SPECIAL_KEYS.clear()
        SPECIAL_KEYS.update(special_keys)
        self.selected_input_device = None
        self.controller = None
        self._engine_thread = None

        self.command_var = tk.StringVar()
        self.device_var = tk.StringVar()
        self.confidence_var = tk.DoubleVar(value=self.config.confidence_threshold)
        self.status_var = tk.StringVar(value="Loading...")
        self.metrics_var = tk.StringVar(value="Inference: -- ms")
        self.mic_status_var = tk.StringVar(value="Microphone off")

        self.command_dropdown = None
        self.mic_dropdown = None
        self.key_entry = None
        self.tree = None
        self.start_btn = None
        self.log_text = None

        self.controller = VoiceController(self.config)

        self._build_ui()
        self._add_default_bindings()
        self._monkey_patch_controller()
        self.confidence_var.trace_add("write", self._sync_confidence)
        self.load_input_devices()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.status_var.set("Ready — press Start Commands")

    def _monkey_patch_controller(self):
        """
        Replace _press_key and the stdout writes in VoiceController
        with pynput key presses and UI callbacks.
        """
        assert self.controller is not None
        controller = self.controller

        def patched_press_key(key_name):
            parsed = self.commands.get(key_name)
            if parsed is None:
                parsed = parse_key(key_name)
            if parsed is not None:
                self.keyboard.press(parsed)
                self.keyboard.release(parsed)

        controller._press_key = patched_press_key

        def patched_classify(waveform, t_capture):
            import time
            import torch
            from audio_processing import preprocess, SAMPLE_RATE

            mel = preprocess(waveform, SAMPLE_RATE, controller.mel_transform)
            mel = mel.unsqueeze(0).to(controller.device)

            with torch.no_grad():
                logits = controller.model(mel)
                probs = torch.softmax(logits, dim=1)[0]

            best_idx = int(probs.argmax().item())
            best_label = controller.labels[best_idx]
            best_prob = probs[best_idx].item()

            if best_label.startswith("_"):
                controller._prev_prediction = None
                controller._streak = 0
                return

            if best_label == controller._prev_prediction:
                controller._streak += 1
            else:
                controller._prev_prediction = best_label
                controller._streak = 1

            self.root.after(
                0, self.status_var.set, f"Heard: {best_label} ({best_prob:.2f})"
            )

            needed = 1 if best_prob >= 0.99 else 2
            if (
                controller._streak >= needed
                and best_prob >= controller.config.confidence_threshold
            ):
                now = time.time()
                if now - controller._last_press_time >= controller.config.cooldown_sec:
                    key = controller.config.key_map.get(best_label)
                    if key:
                        latency_ms = (time.perf_counter() - t_capture) * 1000
                        controller._last_press_time = now
                        controller._streak = 0
                        controller._prev_prediction = None
                        controller._buffer.clear()
                        controller._samples_since_last_classify = 0

                        patched_press_key(best_label)

                        self.root.after(
                            0,
                            self.metrics_var.set,
                            f"Inference: {latency_ms:.1f} ms | Conf: {best_prob:.3f}",
                        )
                        self.root.after(
                            0,
                            self._log,
                            f"[{best_label}] conf={best_prob:.3f} latency={latency_ms:.1f}ms",
                        )
                        self.root.after(
                            0, self.status_var.set, f"Detected: {best_label}"
                        )

        controller._classify = patched_classify

    def _build_ui(self):
        """Construct all tkinter widgets."""
        assert self.controller is not None
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        mic_frame = ttk.LabelFrame(frame, text="Microphone")
        mic_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        mic_frame.columnconfigure(1, weight=1)

        ttk.Label(mic_frame, text="Input Device:").grid(row=0, column=0, padx=5, pady=5)
        self.mic_dropdown = ttk.Combobox(
            mic_frame, textvariable=self.device_var, width=28, state="readonly"
        )
        self.mic_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.mic_dropdown.bind("<<ComboboxSelected>>", self._on_device_changed)
        ttk.Button(
            mic_frame,
            text="Refresh",
            command=self.load_input_devices,
        ).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(mic_frame, textvariable=self.mic_status_var).grid(
            row=1, column=1, columnspan=2, padx=5, pady=(0, 5), sticky="w"
        )

        ttk.Label(frame, text="Voice Command").grid(row=1, column=0)
        ttk.Label(frame, text="Key").grid(row=1, column=1)

        voice_options = sorted(
            label for label in self.controller.labels if not label.startswith("_")
        )
        self.command_dropdown = ttk.Combobox(
            frame,
            textvariable=self.command_var,
            values=voice_options,
            width=22,
            state="readonly",
        )
        self.command_dropdown.grid(row=2, column=0, padx=5, pady=5)

        self.key_entry = ttk.Entry(frame, width=10)
        self.key_entry.grid(row=2, column=1, padx=5)

        ttk.Button(frame, text="Add", command=self.add_binding).grid(
            row=2, column=2, padx=5
        )
        ttk.Button(frame, text="Remove", command=self.remove_binding).grid(
            row=2, column=3, padx=5
        )

        self.tree = ttk.Treeview(
            frame, columns=("command", "key"), show="headings", height=10
        )
        self.tree.heading("command", text="Voice Command")
        self.tree.heading("key", text="Key")
        self.tree.column("command", width=180)
        self.tree.column("key", width=100)
        self.tree.grid(row=3, column=0, columnspan=4, pady=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=4, pady=5)

        self.start_btn = ttk.Button(
            btn_frame, text="Start Commands", command=self.toggle_commands
        )
        self.start_btn.pack(side="left", padx=5)

        ttk.Label(btn_frame, text="Threshold:").pack(side="left", padx=(10, 2))
        ttk.Spinbox(
            btn_frame,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.confidence_var,
            width=5,
        ).pack(side="left")

        ttk.Label(frame, textvariable=self.status_var, font=("Helvetica", 11)).grid(
            row=5, column=0, columnspan=4, pady=5
        )
        ttk.Label(frame, textvariable=self.metrics_var, font=("Helvetica", 10)).grid(
            row=6, column=0, columnspan=4, pady=(0, 5)
        )

        self.log_text = tk.Text(frame, height=8, width=58, state="disabled")
        self.log_text.grid(row=7, column=0, columnspan=4, pady=5)

    def _add_default_bindings(self):
        """Populate bindings from config.key_map."""
        assert self.tree is not None
        for command, key_name in self.config.key_map.items():
            parsed = parse_key(key_name)
            if parsed is not None:
                self.commands[command] = parsed
                self.tree.insert("", "end", values=(command, key_name))

    def add_binding(self):
        assert self.key_entry is not None
        assert self.tree is not None
        command = self.command_var.get().strip()
        key_name = self.key_entry.get().strip().lower()
        parsed = parse_key(key_name)
        if not command or parsed is None:
            return
        for item in self.tree.get_children():
            vals = self.tree.item(item)["values"]
            if vals and vals[0] == command:
                self.tree.delete(item)
                break
        self.commands[command] = parsed
        self.tree.insert("", "end", values=(command, key_name))
        self.command_var.set("")
        self.key_entry.delete(0, tk.END)

    def remove_binding(self):
        assert self.tree is not None
        for item in self.tree.selection():
            vals = self.tree.item(item)["values"]
            if vals:
                self.commands.pop(vals[0], None)
            self.tree.delete(item)

    def toggle_commands(self):
        assert self.controller is not None
        if self.controller.running:
            self._stop_engine()
        else:
            self._start_engine()

    def _start_engine(self):
        """Start calibration + listening in a background thread."""
        assert self.controller is not None
        assert self.start_btn is not None
        controller = self.controller
        start_btn = self.start_btn
        start_btn.config(text="Stop Commands")
        self.status_var.set("Calibrating noise — stay quiet...")
        self.mic_status_var.set("Calibrating...")

        def _bg():
            dev = self.selected_input_device
            controller._calibrate_noise(dev)
            self.root.after(0, self.mic_status_var.set, "Listening")
            self.root.after(0, self.status_var.set, "Listening for speech...")

            controller.running = True
            try:
                with sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    dtype="float32",
                    blocksize=controller._chunk_samples,
                    callback=controller._audio_callback,
                    device=dev,
                ):
                    while controller.running:
                        import time

                        time.sleep(0.1)
            except Exception as e:
                self.root.after(0, self.status_var.set, f"Audio error: {e}")
            finally:
                controller.running = False
                self.root.after(0, start_btn.config, {"text": "Start Commands"})
                self.root.after(0, self.mic_status_var.set, "Microphone off")

        self._engine_thread = threading.Thread(target=_bg, daemon=True)
        self._engine_thread.start()

    def _stop_engine(self):
        assert self.controller is not None
        assert self.start_btn is not None
        controller = self.controller
        start_btn = self.start_btn
        controller.running = False
        start_btn.config(text="Start Commands")
        self.status_var.set("Stopped")
        self.mic_status_var.set("Microphone off")

    def _sync_confidence(self, *_):
        try:
            self.config.confidence_threshold = float(self.confidence_var.get())
        except (tk.TclError, ValueError):
            pass

    def _log(self, message):
        assert self.log_text is not None
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def load_input_devices(self):
        assert self.mic_dropdown is not None
        current = self.device_var.get().strip()
        try:
            devices = sd.query_devices()
        except Exception:
            return

        labels = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                labels.append(f"{i}: {d['name']}")

        self.mic_dropdown["values"] = labels
        if not labels:
            return

        dd = sd.default.device
        default_input = dd[0] if isinstance(dd, (list, tuple)) and dd else None

        selected = current if current in labels else None
        if selected is None and default_input is not None and default_input >= 0:
            selected = next(
                (l for l in labels if l.startswith(f"{default_input}:")), None
            )
        if selected is None:
            selected = labels[0]

        self.device_var.set(selected)
        self.selected_input_device = int(selected.split(":", 1)[0])

    def _on_device_changed(self, _event=None):
        assert self.controller is not None
        selected = self.device_var.get().strip()
        if not selected:
            return
        was_running = self.controller.running
        if was_running:
            self._stop_engine()
        self.selected_input_device = int(selected.split(":", 1)[0])
        if was_running:
            self._start_engine()

    def _on_close(self):
        assert self.controller is not None
        self.controller.running = False
        self.root.destroy()


def main():
    try:
        app_root = tk.Tk()
        VoiceCommandApp(app_root)
        app_root.mainloop()
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
