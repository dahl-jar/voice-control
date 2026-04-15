"""
Download dataset and train the voice command model.

Run: python download_model.py

This will:
  1. Download the Google Speech Commands v2 dataset (~2.3 GB)
  2. Train the VoiceCommandCNN model
  3. Save the trained model to ./models/voice_command_model.pt
"""

import logging
import os
import sys

from log_config import configure_logging


logger = logging.getLogger(__name__)


def check_dependencies():
    missing = []
    for pkg in ["torch", "torchaudio", "sounddevice", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        sys.exit(1)


def check_model_exists():
    from config import TrainConfig
    config = TrainConfig()
    if os.path.exists(config.model_path):
        logger.info(f"Model already exists at {config.model_path}")
        answer = input("Re-train and overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            logger.info("Skipping. Use the existing model.")
            sys.exit(0)


def download_dataset():
    """Downloads Google Speech Commands v2. Importing the dataset triggers the download."""
    from config import TrainConfig
    config = TrainConfig()
    logger.info(f"Downloading Google Speech Commands v2 to {config.data_dir}/...")
    os.makedirs(config.data_dir, exist_ok=True)

    if os.listdir(config.data_dir):
        logger.warning(
            f"{config.data_dir}/ is not empty — torchaudio will reuse whatever's "
            "already there without verifying the checksum. If you suspect the "
            "dataset is partially downloaded or corrupted, delete the directory "
            "and rerun this script."
        )

    import torchaudio
    torchaudio.datasets.SPEECHCOMMANDS(root=config.data_dir, download=True, subset="training")
    logger.info("Dataset downloaded.")


def train_model():
    logger.info("Starting training...")
    from train import train
    train()


def main():
    configure_logging()
    logger.info("=" * 50)
    logger.info("  Voice Command Model Setup")
    logger.info("=" * 50)

    check_dependencies()
    check_model_exists()
    download_dataset()
    train_model()

    from config import TrainConfig
    config = TrainConfig()
    if os.path.exists(config.model_path):
        size_mb = os.path.getsize(config.model_path) / (1024 * 1024)
        logger.info(f"Model saved to {config.model_path} ({size_mb:.1f} MB)")
        logger.info("You can now run: python inference.py")
    else:
        logger.error("Training finished but no model was saved. Check for errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
