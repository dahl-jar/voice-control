# Voice Control

Real-time voice command recognition that maps spoken words to keyboard inputs. Say "up", "down", "left", or "right" and the system presses the corresponding arrow key.

Built on a small CNN trained on Google Speech Commands v2. 

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model isn't included in the repo (too large for git). Train it:

```bash
python download_model.py
```

This downloads the Google Speech Commands v2 dataset (~2.3 GB) and trains for 30 epochs. Takes about 30-40 minutes on CPU.

## Usage

```bash
python inference.py
```

**With GUI:**

```bash
python ui.py
```

The GUI lets you pick your mic, set the confidence threshold, and bind voice commands to keys.

**Debug mode** (shows audio levels and predictions):

```bash
python inference.py --debug
```

## How it works

1. Audio comes in from the mic in 50ms chunks
2. A sliding 750ms window gets classified every 75ms
3. The CNN outputs probabilities for each command
4. If the same command is predicted twice in a row above the confidence threshold, the key fires
5. The audio buffer clears after each fire to prevent double-triggers

The model is a 4-block CNN (~250k parameters) that takes 40-bin log mel spectrograms as input.

## Fine-tuning

If the model doesn't respond well to your voice, you can fine-tune it on personal recordings:

```bash
python finetune.py --record   # record samples
python finetune.py --train    # fine-tune the model
```

## Performance

On the validation set (Google Speech Commands v2):

- Overall accuracy: ~96%
- Inference: ~5ms per prediction
- Commands (up/down/left/right): 94-98% per-class accuracy

Run `performance.ipynb` for the full breakdown with plots.
