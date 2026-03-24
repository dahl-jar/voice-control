"""
Single source of truth for ALL audio preprocessing.
Both training and inference MUST use these functions.
This eliminates train/inference mismatch.
"""

import torch
import torchaudio
import torchaudio.transforms as T

"""
Audio format constants. Changing these requires retraining.
Computed output shape: (1, N_MELS, time_steps) where time_steps = NUM_SAMPLES // HOP_LENGTH + 1 = 101.
"""
SAMPLE_RATE = 16000
DURATION_SEC = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160


def get_mel_transform() -> T.MelSpectrogram:
    """Returns the mel spectrogram transform. Use this everywhere."""
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )


def pad_or_trim(waveform: torch.Tensor) -> torch.Tensor:
    """Pad or trim waveform to exactly NUM_SAMPLES. Works on any shape (..., time)."""
    length = waveform.shape[-1]
    if length > NUM_SAMPLES:
        waveform = waveform[..., :NUM_SAMPLES]
    elif length < NUM_SAMPLES:
        pad_amount = NUM_SAMPLES - length
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    return waveform


def preprocess(waveform: torch.Tensor, sample_rate: int,
               mel_transform: T.MelSpectrogram) -> torch.Tensor:
    """
    THE preprocessing function. Used identically in training and inference.

    Ensures 2D mono input, resamples if needed, normalizes amplitude,
    pads/trims to fixed length, then computes log mel spectrogram
    (with small epsilon to avoid log(0)).

    @param waveform: Raw audio tensor, shape (channels, time) or (time,).
    @param sample_rate: Original sample rate of the audio.
    @param mel_transform: From get_mel_transform().
    @returns: Log mel spectrogram, shape (1, N_MELS, time_steps).
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    waveform = pad_or_trim(waveform)

    mel = mel_transform(waveform)

    mel = torch.log(mel + 1e-9)

    return mel
