# Voice Control

Voice Control is a hands-free keyboard. It listens to your microphone in real time, and when you say "up", "down", "left", or "right", it presses the matching arrow key on your computer. You can use it to control games, scroll pages, navigate menus, or anything else that reacts to arrow keys — without touching the keyboard.

Under the hood it runs a small convolutional neural network trained on Google Speech Commands v2. Audio is captured from the mic, chopped into short sliding windows, and classified locally on your machine. Nothing is sent to the cloud. A simple GUI lets you pick your microphone, tune the confidence threshold, and rebind commands to different keys.

## Requirements

- Python 3.12
- A working microphone

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The trained model is already in `models/`, so you can run it straight away.

- **macOS**: works as-is. You'll be asked to grant microphone and accessibility permissions the first time.
- **Windows**: works as-is. Use PowerShell for the activate command shown above.
- **Linux**: the GUI does not work — use the CLI instead. Install system packages first; on Arch: `sudo pacman -S python tk portaudio`. Pure Wayland sessions may not allow global key injection — use X11/Xwayland if keys don't fire.

## Run

```bash
python -m voice_control.runtime.inference   # CLI
python -m voice_control.runtime.ui          # GUI
```

To retrain from scratch: `python scripts/download_model.py`.
