"""
Feature extraction shared by train and inference — both must call
`preprocess` to avoid silent train/serve skew.

Constants are baked into the checkpoint; changing them invalidates
the model. Output shape: (1, N_MELS, 101).
"""

import torch
import torchaudio
import torchaudio.transforms as T


SAMPLE_RATE = 16000
DURATION_SEC = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160


def get_mel_transform() -> T.MelSpectrogram:
    """Build the MelSpectrogram op used by train and inference."""
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )


def pad_or_trim(waveform: torch.Tensor) -> torch.Tensor:
    """Force waveform length to NUM_SAMPLES on the last dim."""
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
    Raw waveform -> log mel spectrogram. Must stay byte-identical
    between train and inference — retrain if you touch this.

    Log uses a 1e-9 floor so silent frames don't go to -inf.

    @param waveform: Raw audio, (C, T) or (T,).
    @param sample_rate: Source rate. Resampled to SAMPLE_RATE if mismatched.
    @param mel_transform: Instance from `get_mel_transform()`.
    @returns: Log mel spectrogram, (1, N_MELS, 101).
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
