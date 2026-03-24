"""
Training and inference configuration. Edit this to tune the system.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    """
    Training configuration.

    @param commands: Commands to recognize. These map to key presses.
    @param epochs: Number of training epochs.
    @param batch_size: Batch size for training.
    @param learning_rate: Initial learning rate.
    @param weight_decay: Weight decay for optimizer.
    @param lr_scheduler_patience: Patience for learning rate scheduler.
    @param lr_scheduler_factor: Factor for learning rate reduction.
    @param augment_noise: Whether to add noise augmentation (training only, never inference).
    @param noise_snr_db: Signal-to-noise ratio in dB for noise augmentation.
    @param augment_time_shift: Whether to apply time shift augmentation (training only).
    @param time_shift_max_ms: Maximum time shift in milliseconds.
    @param augment_speed: Whether to apply speed perturbation (training only).
    @param speed_range: Min and max speed factors.
    @param data_dir: Directory for the dataset.
    @param val_split: Fraction of data used for validation.
    @param num_workers: Number of dataloader workers.
    @param model_path: Path to save the trained model.
    """
    commands: List[str] = field(default_factory=lambda: [
        "up", "down", "left", "right",
    ])

    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5

    augment_noise: bool = True
    noise_snr_db: float = 10.0
    augment_time_shift: bool = True
    time_shift_max_ms: int = 100
    augment_speed: bool = True
    speed_range: tuple = (0.9, 1.1)

    data_dir: str = "./data"
    val_split: float = 0.2
    num_workers: int = 4

    model_path: str = "./models/voice_command_model.pt"


@dataclass
class InferenceConfig:
    """
    Inference configuration.

    @param model_path: Path to the trained model checkpoint.
    @param confidence_threshold: Predictions below this are ignored.
        Higher = fewer false positives but might miss quiet commands.
        Lower = more responsive but more false triggers.
    @param chunk_duration_sec: Duration of each audio chunk captured.
    @param window_duration_sec: Sliding window duration for classification.
    @param stride_duration_sec: Stride between classification windows.
    @param cooldown_sec: Minimum seconds between key presses.
    @param key_map: Mapping from voice command to keyboard key.
    @param device: Torch device for inference.
    """
    model_path: str = "./models/voice_command_model.pt"

    confidence_threshold: float = 0.93

    chunk_duration_sec: float = 0.05
    window_duration_sec: float = 0.75
    stride_duration_sec: float = 0.075

    cooldown_sec: float = 0.5

    key_map: dict = field(default_factory=lambda: {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
    })

    device: str = "cuda"
