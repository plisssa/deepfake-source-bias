"""Шумовые аугментации (восстановлены из legacy COL-2 NoiseAugmentor).

Семантика как в legacy/COL-2/main.py: gaussian-шум добавляется в нормированном
[0, 1] пространстве, poisson-шум моделирует фотонный шум через
``poisson(img * scale) / scale``. Все функции принимают и возвращают uint8;
детерминизм — через переданный ``np.random.Generator``.
"""

from __future__ import annotations

import numpy as np


def _get_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def gaussian_noise(
    img: np.ndarray, sigma: float = 8.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Аддитивный гауссов шум со std=sigma в шкале [0, 255].

    Эквивалент gaussian-части NoiseAugmentor из COL-2 (там std задавался в [0, 1]:
    gaussian_std=0.02 соответствует sigma≈5.1 здесь).
    """
    rng = _get_rng(rng)
    noisy = img.astype(np.float32) + rng.normal(0.0, float(sigma), img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def poisson_noise(
    img: np.ndarray, rng: np.random.Generator | None = None, scale: float = 30.0
) -> np.ndarray:
    """Пуассоновский (фотонный) шум: poisson(img * scale) / scale, как в COL-2."""
    rng = _get_rng(rng)
    img_f = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
    noisy = rng.poisson(img_f * float(scale)).astype(np.float32) / float(scale)
    return (np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8)


def gaussian_poisson(
    img: np.ndarray,
    sigma: float = 5.0,
    rng: np.random.Generator | None = None,
    scale: float = 30.0,
) -> np.ndarray:
    """Композиция gaussian → poisson — точная семантика NoiseAugmentor.augment (COL-2)."""
    rng = _get_rng(rng)
    img_f = img.astype(np.float32) / 255.0
    img_f = img_f + rng.normal(0.0, float(sigma) / 255.0, img_f.shape)
    noisy = rng.poisson(np.clip(img_f, 0.0, 1.0) * float(scale)).astype(np.float32)
    noisy = noisy / float(scale)
    return (np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8)
