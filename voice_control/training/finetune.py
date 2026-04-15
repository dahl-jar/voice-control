"""
Fine-tune the base model on recordings from one speaker.

  --record : captures WAVs into recordings/<command>/
  --train  : short low-LR pass from the base checkpoint

LR stays well below base training LR to avoid catastrophic forgetting.
"""

import logging
import os
import sys
import random

import torch
import torch.nn as nn
import sounddevice as sd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from voice_control.audio.io import load_waveform, save_waveform
from voice_control.audio.processing import preprocess, get_mel_transform, SAMPLE_RATE, NUM_SAMPLES
from voice_control.log_config import configure_logging
from voice_control.model import VoiceCommandCNN
from voice_control.config import InferenceConfig, TrainConfig, repo_path


logger = logging.getLogger(__name__)


RECORDINGS_DIR = repo_path("recordings")


def record_samples():
    """
    Interactive recording session, one command at a time. Resumes from
    whatever's already on disk. Enter-gated (not auto-triggered) so
    speaker pacing doesn't mess with the window alignment.
    """
    config = TrainConfig()
    commands = sorted(config.commands)

    logger.info("=== Voice Recording Session ===")
    logger.info(f"Commands to record: {commands}")
    logger.info("You'll record each command multiple times.")
    logger.info(f"Recordings saved to: {RECORDINGS_DIR}/")

    samples_per_command = 50
    logger.info(f"Target: {samples_per_command} recordings per command")
    logger.info("Press Enter to start recording, speak the command, wait 1 second.")

    for cmd in commands:
        cmd_dir = os.path.join(RECORDINGS_DIR, cmd)
        os.makedirs(cmd_dir, exist_ok=True)

        existing = len([f for f in os.listdir(cmd_dir) if f.endswith(".wav")])
        logger.info(f"--- Command: '{cmd}' ({existing} existing) ---")

        for i in range(existing, samples_per_command):
            input(f"  [{i + 1}/{samples_per_command}] Press Enter, then say '{cmd}': ")
            logger.info("Recording 1 second...")

            audio = sd.rec(
                NUM_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="float32"
            )
            sd.wait()
            logger.info("done.")

            filepath = os.path.join(cmd_dir, f"{cmd}_{i:04d}.wav")
            save_waveform(filepath, audio, SAMPLE_RATE)

    logger.info(f"Recording complete! Files in {RECORDINGS_DIR}/")
    logger.info("Run: python finetune.py --train")


class RecordingsDataset(Dataset):
    """Dataset from your recorded samples."""

    def __init__(self, labels: list, augment: bool = False):
        self.labels = labels
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.mel_transform = get_mel_transform()
        self.augment = augment
        self.samples = []

        for label in labels:
            if label.startswith("_"):
                continue
            cmd_dir = os.path.join(RECORDINGS_DIR, label)
            if not os.path.isdir(cmd_dir):
                continue
            for f in os.listdir(cmd_dir):
                if f.endswith(".wav"):
                    self.samples.append(
                        (os.path.join(cmd_dir, f), self.label_to_idx[label])
                    )

        logger.info(f"Loaded {len(self.samples)} recordings")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns (mel_spectrogram, label_index). Applies light noise and time shift
        augmentation when enabled.
        """
        path, label_idx = self.samples[idx]
        waveform, sr = load_waveform(path)

        if self.augment:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            shift = random.randint(-1600, 1600)
            waveform = torch.roll(waveform, shift, dims=-1)

        mel = preprocess(waveform, sr, self.mel_transform)
        return mel, label_idx


def finetune():
    """
    Short fine-tune from the base checkpoint on local recordings.
    LR=1e-4 (1/10 of base), 20 epochs. Saves as `*_finetuned.pt` so
    the base checkpoint stays intact as a fallback.
    """
    config = InferenceConfig()

    checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoiceCommandCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded base model (val_acc={checkpoint['val_acc']:.4f})")

    dataset = RecordingsDataset(labels, augment=True)
    if len(dataset) == 0:
        logger.error("No recordings found! Run: python finetune.py --record")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Fine-tuning for 20 epochs on {len(dataset)} samples...")
    acc = 0.0
    for epoch in range(20):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for mel, target in loader:
            mel, target = mel.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)

        acc = correct / total
        logger.info(
            f"  Epoch {epoch + 1}/20: loss={total_loss / len(loader):.4f} acc={acc:.4f}"
        )

    save_path = config.model_path.replace(".pt", "_finetuned.pt")
    save_data = {
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "num_classes": len(labels),
        "val_acc": acc,
        "epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
        "finetuned": True,
    }
    torch.save(save_data, save_path)
    logger.info(f"Fine-tuned model saved to: {save_path}")
    logger.info("To use it, update model_path in config.py or run:")
    logger.info("  python inference.py  (after updating config.py)")


if __name__ == "__main__":
    configure_logging()
    if "--record" in sys.argv:
        record_samples()
    elif "--train" in sys.argv:
        finetune()
    else:
        logger.info("Usage:")
        logger.info("  python finetune.py --record   Record your voice samples")
        logger.info("  python finetune.py --train    Fine-tune model on recordings")
