"""Shared WAV I/O helpers for the project."""

import numpy as np
import soundfile as sf
import torch


def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    """Load a WAV file as a torch tensor without relying on TorchCodec."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))
    return waveform, sample_rate


def save_waveform(
    path: str, waveform: np.ndarray | torch.Tensor, sample_rate: int
) -> None:
    """Save mono or multi-channel waveform data as WAV."""
    if isinstance(waveform, torch.Tensor):
        samples = waveform.detach().cpu().numpy()
        if samples.ndim == 2:
            samples = samples.T
    else:
        samples = waveform

    sf.write(path, samples, sample_rate)
