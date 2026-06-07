"""
Audio augmentation for Tamil TTS training.

Applies subtle augmentations that improve model robustness
without degrading audio quality:
  - Speed perturbation (0.9x - 1.1x)
  - Volume normalization
  - Silence trimming
  - Optional noise injection (very light)
"""

import random
from typing import Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as F


def trim_silence(
    waveform: torch.Tensor,
    sr: int = 24000,
    threshold_db: float = -40.0,
    min_silence_ms: float = 100,
) -> torch.Tensor:
    """Remove leading/trailing silence from audio."""
    frame_size = int(sr * min_silence_ms / 1000)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    energy = waveform.pow(2).unfold(-1, frame_size, frame_size // 2).mean(dim=-1)
    threshold = 10 ** (threshold_db / 10)

    above_threshold = energy.squeeze() > threshold
    if not above_threshold.any():
        return waveform

    indices = torch.where(above_threshold)[0]
    start = max(0, indices[0].item() * (frame_size // 2) - frame_size)
    end = min(waveform.shape[-1], (indices[-1].item() + 1) * (frame_size // 2) + frame_size)

    return waveform[..., start:end]


def normalize_volume(
    waveform: torch.Tensor,
    target_db: float = -20.0,
) -> torch.Tensor:
    """Normalize audio to target RMS level."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-8:
        return waveform

    current_db = 20 * torch.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    return (waveform * gain).clamp(-1.0, 1.0)


def speed_perturbation(
    waveform: torch.Tensor,
    sr: int = 24000,
    factor: Optional[float] = None,
    min_factor: float = 0.9,
    max_factor: float = 1.1,
) -> Tuple[torch.Tensor, int]:
    """Apply speed perturbation (changes pitch too)."""
    if factor is None:
        factor = random.uniform(min_factor, max_factor)

    if abs(factor - 1.0) < 0.01:
        return waveform, sr

    effects = [["speed", str(factor)], ["rate", str(sr)]]
    augmented, new_sr = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
    return augmented, new_sr


def add_light_noise(
    waveform: torch.Tensor,
    snr_db: float = 40.0,
) -> torch.Tensor:
    """Add very light Gaussian noise (for robustness, not degradation)."""
    noise = torch.randn_like(waveform)
    signal_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean()

    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (noise_power * snr_linear))

    return waveform + noise * scale


def augment_audio(
    waveform: torch.Tensor,
    sr: int = 24000,
    trim: bool = True,
    normalize: bool = True,
    speed_perturb: bool = True,
    speed_prob: float = 0.3,
    noise: bool = False,
    noise_prob: float = 0.1,
) -> Tuple[torch.Tensor, int]:
    """Apply augmentation pipeline to a single audio sample."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if trim:
        waveform = trim_silence(waveform, sr)

    if normalize:
        waveform = normalize_volume(waveform)

    if speed_perturb and random.random() < speed_prob:
        waveform, sr = speed_perturbation(waveform, sr)

    if noise and random.random() < noise_prob:
        waveform = add_light_noise(waveform, snr_db=40.0)

    return waveform, sr
