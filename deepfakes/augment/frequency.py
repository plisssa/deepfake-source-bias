"""Частотные аугментации (восстановлены из legacy GRAD-4 FrequencyAugmentor).

Семантика как в legacy/GRAD-4/main.py: амплитуда FFT умножается на общий
случайный коэффициент 1 + U(-strength, strength) при сохранении фазы.
Принимает и возвращает uint8 (grayscale или RGB — по каналам, один scale).
"""

from __future__ import annotations

import numpy as np


def _jitter_channel(channel: np.ndarray, scale: float) -> np.ndarray:
    fft = np.fft.fftshift(np.fft.fft2(channel.astype(np.float32)))
    amp = np.abs(fft)
    phase = np.angle(fft)
    fft2 = amp * scale * np.exp(1j * phase)
    return np.abs(np.fft.ifft2(np.fft.ifftshift(fft2)))


def fft_amplitude_jitter(
    img: np.ndarray, strength: float = 0.2, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Jitter амплитуды FFT: amp *= 1 + U(-strength, strength), фаза сохраняется.

    Один общий scale на изображение (как в GRAD-4); для цветных изображений
    он применяется к каждому каналу одинаково.
    """
    rng = rng if rng is not None else np.random.default_rng()
    scale = 1.0 + float(rng.uniform(-float(strength), float(strength)))
    if img.ndim == 2:
        out = _jitter_channel(img, scale)
    else:
        out = np.stack([_jitter_channel(img[..., c], scale) for c in range(img.shape[-1])], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)
