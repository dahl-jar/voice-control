"""
Download dataset and train the voice command model.

Run: python download_model.py

This will:
  1. Download the Google Speech Commands v2 dataset (~2.3 GB)
  2. Train the VoiceCommandCNN model
  3. Save the trained model to ./models/voice_command_model.pt
"""

import os
import sys


def check_dependencies():
    missing = []
    for pkg in ["torch", "torchaudio", "sounddevice", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)


def check_model_exists():
    from config import TrainConfig
    config = TrainConfig()
    if os.path.exists(config.model_path):
        print(f"Model already exists at {config.model_path}")
        answer = input("Re-train and overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipping. Use the existing model.")
            sys.exit(0)


def download_dataset():
    """Downloads Google Speech Commands v2. Importing the dataset triggers the download."""
    from config import TrainConfig
    config = TrainConfig()
    print(f"Downloading Google Speech Commands v2 to {config.data_dir}/...")
    os.makedirs(config.data_dir, exist_ok=True)
    import torchaudio
    torchaudio.datasets.SPEECHCOMMANDS(root=config.data_dir, download=True, subset="training")
    print("Dataset downloaded.")


def train_model():
    print("\nStarting training...\n")
    from train import train
    train()


def main():
    print("=" * 50)
    print("  Voice Command Model Setup")
    print("=" * 50)
    print()

    check_dependencies()
    check_model_exists()
    download_dataset()
    train_model()

    from config import TrainConfig
    config = TrainConfig()
    if os.path.exists(config.model_path):
        size_mb = os.path.getsize(config.model_path) / (1024 * 1024)
        print(f"\nModel saved to {config.model_path} ({size_mb:.1f} MB)")
        print("You can now run: python inference.py")
    else:
        print("\nTraining finished but no model was saved. Check for errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
