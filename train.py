"""
Training script. Run: python train.py
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import TrainConfig
from model import VoiceCommandCNN
from dataset import create_dataloaders
from audio_processing import get_mel_transform, SAMPLE_RATE, N_MELS


def train():
    """
    Train the voice command model.

    Labels are: sorted commands + _unknown + _silence.
    Uses AdamW optimizer with ReduceLROnPlateau scheduler.
    Saves the best model checkpoint by validation accuracy.
    """
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels = sorted(config.commands) + ["_unknown", "_silence"]
    num_classes = len(labels)
    print(f"Classes ({num_classes}): {labels}")

    print("Loading dataset (downloads on first run)...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = VoiceCommandCNN(num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                   patience=config.lr_scheduler_patience,
                                   factor=config.lr_scheduler_factor)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (mel, target) in enumerate(train_loader):
            mel, target = mel.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{config.epochs} "
                      f"[{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}")

        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        per_class_correct = [0] * num_classes
        per_class_total = [0] * num_classes

        with torch.no_grad():
            for mel, target in val_loader:
                mel, target = mel.to(device), target.to(device)
                output = model(mel)
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)

                for i in range(num_classes):
                    mask = target == i
                    per_class_correct[i] += (pred[mask] == i).sum().item()
                    per_class_total[i] += mask.sum().item()

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(f"\nEpoch {epoch+1}/{config.epochs}: "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        for i, label in enumerate(labels):
            if per_class_total[i] > 0:
                acc = per_class_correct[i] / per_class_total[i]
                print(f"  {label:>10s}: {acc:.4f} ({per_class_correct[i]}/{per_class_total[i]})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_data = {
                "model_state_dict": model.state_dict(),
                "labels": labels,
                "num_classes": num_classes,
                "val_acc": val_acc,
                "epoch": epoch + 1,
                "config": {
                    "sample_rate": SAMPLE_RATE,
                    "n_mels": N_MELS,
                },
            }
            torch.save(save_data, config.model_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

        print()

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Model saved to: {config.model_path}")


if __name__ == "__main__":
    train()
