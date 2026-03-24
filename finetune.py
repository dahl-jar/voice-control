"""
Fine-tune on YOUR voice recordings for better accuracy.

Usage:
  1. Record samples:  python finetune.py --record
  2. Fine-tune model:  python finetune.py --train
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torchaudio
import sounddevice as sd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from audio_processing import preprocess, get_mel_transform, SAMPLE_RATE, NUM_SAMPLES
from model import VoiceCommandCNN
from config import TrainConfig, InferenceConfig


RECORDINGS_DIR = "./recordings"


def record_samples():
    """Interactive recording session for each command."""
    config = TrainConfig()
    commands = sorted(config.commands)

    print("=== Voice Recording Session ===")
    print(f"Commands to record: {commands}")
    print(f"You'll record each command multiple times.")
    print(f"Recordings saved to: {RECORDINGS_DIR}/\n")

    samples_per_command = 50
    print(f"Target: {samples_per_command} recordings per command")
    print("Press Enter to start recording, speak the command, wait 1 second.\n")

    for cmd in commands:
        cmd_dir = os.path.join(RECORDINGS_DIR, cmd)
        os.makedirs(cmd_dir, exist_ok=True)

        existing = len([f for f in os.listdir(cmd_dir) if f.endswith(".wav")])
        print(f"\n--- Command: '{cmd}' ({existing} existing) ---")

        for i in range(existing, samples_per_command):
            input(f"  [{i+1}/{samples_per_command}] Press Enter, then say '{cmd}': ")
            print("    Recording 1 second...", end="", flush=True)

            audio = sd.rec(NUM_SAMPLES, samplerate=SAMPLE_RATE,
                          channels=1, dtype="float32")
            sd.wait()
            print(" done.")

            filepath = os.path.join(cmd_dir, f"{cmd}_{i:04d}.wav")
            waveform = torch.from_numpy(audio.T)
            torchaudio.save(filepath, waveform, SAMPLE_RATE)

    print(f"\nRecording complete! Files in {RECORDINGS_DIR}/")
    print("Run: python finetune.py --train")


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
                    self.samples.append((os.path.join(cmd_dir, f),
                                        self.label_to_idx[label]))

        print(f"Loaded {len(self.samples)} recordings")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns (mel_spectrogram, label_index). Applies light noise and time shift
        augmentation when enabled.
        """
        path, label_idx = self.samples[idx]
        waveform, sr = torchaudio.load(path)

        if self.augment:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            shift = random.randint(-1600, 1600)
            waveform = torch.roll(waveform, shift, dims=-1)

        mel = preprocess(waveform, sr, self.mel_transform)
        return mel, label_idx


def finetune():
    """
    Fine-tune the base model on your recordings.

    Loads the base checkpoint, creates a dataset from recorded samples,
    and fine-tunes with a low learning rate to preserve base knowledge.
    """
    config = InferenceConfig()

    checkpoint = torch.load(config.model_path, map_location="cpu",
                            weights_only=True)
    labels = checkpoint["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoiceCommandCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded base model (val_acc={checkpoint['val_acc']:.4f})")

    dataset = RecordingsDataset(labels, augment=True)
    if len(dataset) == 0:
        print("No recordings found! Run: python finetune.py --record")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\nFine-tuning for 20 epochs on {len(dataset)} samples...")
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
        print(f"  Epoch {epoch+1}/20: loss={total_loss/len(loader):.4f} acc={acc:.4f}")

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
    print(f"\nFine-tuned model saved to: {save_path}")
    print(f"To use it, update model_path in config.py or run:")
    print(f"  python inference.py  (after updating config.py)")


if __name__ == "__main__":
    if "--record" in sys.argv:
        record_samples()
    elif "--train" in sys.argv:
        finetune()
    else:
        print("Usage:")
        print("  python finetune.py --record   Record your voice samples")
        print("  python finetune.py --train    Fine-tune model on recordings")
