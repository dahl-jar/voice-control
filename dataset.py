"""
Dataset loader for Google Speech Commands v2.
Downloads automatically on first run.
"""

import os
import random
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset

from audio_io import load_waveform
from audio_processing import preprocess, get_mel_transform, SAMPLE_RATE, NUM_SAMPLES
from config import TrainConfig


class SpeechCommandsDataset(Dataset):
    """Wraps torchaudio's SpeechCommands with our preprocessing."""

    def __init__(self, config: TrainConfig, augment: bool = False):
        """
        Downloads and loads Google Speech Commands, filtering and balancing classes.

        Labels are: sorted commands + "_unknown" + "_silence".
        Unknowns are subsampled to balance with the largest command class.
        Synthetic silence samples are added at the same count.

        @param config: Training configuration.
        @param augment: Whether to apply data augmentation.
        """
        self.config = config
        self.augment = augment
        self.mel_transform = get_mel_transform()

        self.labels = sorted(config.commands) + ["_unknown", "_silence"]
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.command_set = set(config.commands)

        os.makedirs(config.data_dir, exist_ok=True)
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=config.data_dir,
            download=True,
            subset="training",
        )

        self.samples: List[Tuple[int, int]] = []
        unknown_indices = []

        for i in range(len(self.dataset)):
            filepath = self.dataset._walker[i]
            label = os.path.basename(os.path.dirname(filepath))

            if label in self.command_set:
                self.samples.append((i, self.label_to_idx[label]))
            elif label != "_background_noise_":
                unknown_indices.append(i)

        max_per_class = (
            max(
                sum(1 for _, l in self.samples if l == idx)
                for idx in range(len(config.commands))
            )
            if self.samples
            else 1000
        )
        random.shuffle(unknown_indices)
        unknown_idx = self.label_to_idx["_unknown"]
        for i in unknown_indices[:max_per_class]:
            self.samples.append((i, unknown_idx))

        silence_idx = self.label_to_idx["_silence"]
        for _ in range(max_per_class):
            self.samples.append((-1, silence_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns (mel_spectrogram, label_index) for the given sample.

        Silence samples (dataset_idx == -1) are generated as small random noise.
        Augmentation is applied only during training. Uses the same preprocessing
        pipeline as inference.
        """
        dataset_idx, label_idx = self.samples[idx]

        if dataset_idx == -1:
            waveform = torch.randn(1, NUM_SAMPLES) * 0.01
            sample_rate = SAMPLE_RATE
        else:
            path = self.dataset._walker[dataset_idx]
            waveform, sample_rate = load_waveform(path)

        if self.augment:
            waveform = self._augment(waveform, sample_rate)

        mel = preprocess(waveform, sample_rate, self.mel_transform)

        return mel, label_idx

    def _augment(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Data augmentation applied ONLY during training.

        Applies time shift, additive noise, and speed perturbation via resampling.

        @param waveform: Raw audio tensor.
        @param sample_rate: Sample rate of the waveform.
        @returns: Augmented waveform tensor.
        """
        cfg = self.config

        if cfg.augment_time_shift:
            max_shift = int(cfg.time_shift_max_ms * sample_rate / 1000)
            shift = random.randint(-max_shift, max_shift)
            waveform = torch.roll(waveform, shift, dims=-1)

        if cfg.augment_noise:
            noise = torch.randn_like(waveform)
            signal_power = waveform.pow(2).mean()
            noise_power = noise.pow(2).mean()
            if noise_power > 0:
                snr_linear = 10 ** (cfg.noise_snr_db / 10)
                scale = torch.sqrt(signal_power / (noise_power * snr_linear))
                waveform = waveform + scale * noise

        if cfg.augment_speed:
            speed = random.uniform(*cfg.speed_range)
            orig_len = waveform.shape[-1]
            new_len = int(orig_len / speed)
            if new_len > 0:
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0),
                    size=new_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

        return waveform


def create_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = SpeechCommandsDataset(config, augment=True)
    val_dataset = SpeechCommandsDataset(config, augment=False)

    n = len(train_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - config.val_split))

    train_loader = DataLoader(
        Subset(train_dataset, indices[:split]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(val_dataset, indices[split:]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
