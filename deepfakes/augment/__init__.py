"""Аугментации изображений: реестр AUGMENTERS для универсального хука BasePipeline.

Каждый аугментер — callable(img, rng, **params) -> uint8-изображение.
Семантика восстановлена из legacy-скриптов (COL-2 NoiseAugmentor,
TEX-2 add_texture_noise ≡ gaussian_noise, GRAD-4 FrequencyAugmentor).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from deepfakes.augment.frequency import fft_amplitude_jitter
from deepfakes.augment.noise import gaussian_noise, gaussian_poisson, poisson_noise
from deepfakes.augment.spatial import hflip, random_crop_resize


def _aug_gaussian_noise(img: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray:
    return gaussian_noise(img, rng=rng, **params)


def _aug_poisson_noise(img: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray:
    return poisson_noise(img, rng=rng, **params)


def _aug_gaussian_poisson(img: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray:
    return gaussian_poisson(img, rng=rng, **params)


def _aug_fft_amplitude_jitter(img: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray:
    return fft_amplitude_jitter(img, rng=rng, **params)


def _aug_hflip(img: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray:
    return hflip(img)


def _aug_random_crop_resize(img: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray:
    return random_crop_resize(img, rng=rng, **params)


AUGMENTERS: dict[str, Callable[..., np.ndarray]] = {
    "gaussian_noise": _aug_gaussian_noise,
    "poisson_noise": _aug_poisson_noise,
    "gaussian_poisson": _aug_gaussian_poisson,
    "fft_amplitude_jitter": _aug_fft_amplitude_jitter,
    "hflip": _aug_hflip,
    "random_crop_resize": _aug_random_crop_resize,
}

__all__ = [
    "AUGMENTERS",
    "fft_amplitude_jitter",
    "gaussian_noise",
    "gaussian_poisson",
    "hflip",
    "poisson_noise",
    "random_crop_resize",
]
