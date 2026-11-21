"""Эквивалентность векторизованного ColorMomentsExtractor старой реализации.

Векторизация НЕ должна менять признаки: кэши и сравнимость результатов
между запусками зависят от идентичности значений (в пределах float32)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deepfakes.features.color import (
    ColorMomentsExtractor,
    _channel_ranges,
    _circular_mean_std_hue,
    _patch_stats,
)


def _reference_extract(ex: ColorMomentsExtractor, img_rgb: np.ndarray) -> np.ndarray:
    """Старая (поэлементная) реализация _extract — эталон."""
    from scipy.stats import skew

    img = ex._convert(img_rgb)
    h, w = img.shape[:2]
    channels = cv2.split(img)
    ranges = _channel_ranges(ex.colorspace)
    feats: list[float] = []
    for ps in ex.patch_sizes:
        for y in range(0, h, ps):
            for x in range(0, w, ps):
                if y + ps > h or x + ps > w:
                    continue
                for ci, c in enumerate(channels):
                    patch = c[y : y + ps, x : x + ps]
                    p = patch.reshape(-1).astype(np.float64)
                    if ex.full_stats:
                        stats = _patch_stats(patch, value_range=ranges[ci])
                    else:
                        stats = [
                            float(np.mean(p)),
                            float(np.std(p)),
                            float(skew(p)) if p.std() > 1e-8 else 0.0,
                        ]
                    if ex.colorspace == "HSV" and ci == 0 and p.size > 0:
                        stats[0], stats[1] = _circular_mean_std_hue(p)
                    feats.extend(stats)
    return np.asarray(feats, dtype=np.float32)


@pytest.mark.parametrize("colorspace", ["RGB", "HSV", "YCbCr", "Lab"])
@pytest.mark.parametrize("full_stats", [True, False])
def test_vectorized_matches_reference(colorspace, full_stats):
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    ex = ColorMomentsExtractor(
        img_size=96,
        patch_sizes=(8, 16, 32),
        colorspace=colorspace,
        full_stats=full_stats,
        verbose=False,
    )
    fast = ex._extract(img)
    ref = _reference_extract(ex, img)
    assert fast.shape == ref.shape
    np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-5)


def test_vectorized_flat_image_no_nan():
    """Плоское изображение: std=0 → skew/kurt должны быть 0, не NaN."""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    ex = ColorMomentsExtractor(img_size=64, patch_sizes=(16,), verbose=False)
    f = ex._extract(img)
    assert np.isfinite(f).all()
