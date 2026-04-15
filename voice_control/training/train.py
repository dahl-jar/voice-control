"""
Training loop. Saves best checkpoint by val_acc (not final) because
the LR scheduler can push the model past its peak in later epochs.
"""

import logging
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from voice_control.audio.processing import SAMPLE_RATE, N_MELS
from voice_control.config import TrainConfig
from voice_control.diagnostics import diagnose
from voice_control.log_config import configure_logging
from voice_control.model import VoiceCommandCNN
from voice_control.training.dataset import create_dataloaders


logger = logging.getLogger(__name__)


def train():
    """
    Run the full training loop and save the best checkpoint.

    AdamW (decoupled weight decay), ReduceLROnPlateau on val_acc
    (loss can drop while accuracy stalls). Checkpoint bakes in
    `labels`/`num_classes` so inference can't drift from config.
    """
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    labels = sorted(config.commands) + ["_unknown", "_silence"]
    num_classes = len(labels)
    logger.info(f"Classes ({num_classes}): {labels}")

    logger.info("Loading dataset (downloads on first run)...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = VoiceCommandCNN(num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                   patience=config.lr_scheduler_patience,
                                   factor=config.lr_scheduler_factor)

    best_val_acc = 0.0
    prev_val_acc = 0.0
    consecutive_regressions = 0
    overfit_warn_threshold = 3
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
                logger.info(f"  Epoch {epoch+1}/{config.epochs} "
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

        if val_acc < prev_val_acc:
            consecutive_regressions += 1
        else:
            consecutive_regressions = 0
        prev_val_acc = val_acc

        if consecutive_regressions >= overfit_warn_threshold:
            logger.warning(
                f"val_acc has regressed {consecutive_regressions} epochs in a row "
                f"while train_acc={train_acc:.4f} — possible overfit. The best "
                f"checkpoint at val_acc={best_val_acc:.4f} is still on disk; "
                "consider early-stopping or lowering the LR further."
            )

        logger.info(f"Epoch {epoch+1}/{config.epochs}: "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}")

        for i, label in enumerate(labels):
            if per_class_total[i] > 0:
                acc = per_class_correct[i] / per_class_total[i]
                logger.info(f"  {label:>10s}: {acc:.4f} ({per_class_correct[i]}/{per_class_total[i]})")

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
            logger.info(f"  -> Saved best model (val_acc={val_acc:.4f})")

    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {config.model_path}")


if __name__ == "__main__":
    configure_logging()
    with diagnose("training_total") as training_diag:
        train()
    logger.info(training_diag.format_line())
