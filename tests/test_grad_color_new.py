"""Тесты блока «градиенты + цвет»: ScharrExtractor (GRAD-2), фиксы LoG/Canny и color.py."""

from __future__ import annotations

import numpy as np
import pytest

from deepfakes.features.color import (
    ColorHistogramExtractor,
    ColorMomentsExtractor,
    YCbCrChiSquareExtractor,
    _channel_ranges,
    _circular_mean_std_hue,
    _convert_colorspace,
)
from deepfakes.features.grad import (
    CannyEdgeExtractor,
    LoGExtractor,
    ScharrExtractor,
    _log_zero_crossings,
)

RNG_SEED = 0


def _make_gray(rng: np.random.Generator, fake: bool, size: int = 64) -> np.ndarray:
    """Синтетика: real = шум; fake = шум + шахматка с клеткой 2px."""
    img = rng.normal(128, 20, (size, size))
    if fake:
        yy, xx = np.indices((size, size))
        chess = ((yy // 2 + xx // 2) % 2).astype(np.float64) * 12.0
        img = img + chess
    return img.clip(0, 255).astype(np.uint8)


def _make_rgb(rng: np.random.Generator, fake: bool, size: int = 64) -> np.ndarray:
    g = _make_gray(rng, fake, size)
    return np.stack([g, g, g], axis=-1)


def _separates(extractor, n_per: int = 16) -> bool:
    """Хотя бы один признак разделяет real/fake по средним (относительно разброса)."""
    rng = np.random.default_rng(RNG_SEED)
    make = _make_gray if extractor.grayscale else _make_rgb
    real = np.stack([extractor._extract(make(rng, fake=False)) for _ in range(n_per)])
    fake = np.stack([extractor._extract(make(rng, fake=True)) for _ in range(n_per)])
    diff = np.abs(real.mean(axis=0) - fake.mean(axis=0))
    spread = real.std(axis=0) + fake.std(axis=0) + 1e-9
    return bool((diff > 0.5 * spread).any())


# ---------- ScharrExtractor (GRAD-2) ----------


def test_scharr_shape_and_dtype():
    e = ScharrExtractor(img_size=64, patch_grid=4, n_orient_bins=9, verbose=False)
    rng = np.random.default_rng(RNG_SEED)
    v = e._extract(_make_gray(rng, fake=False))
    assert v.ndim == 1
    assert v.dtype == np.float32
    # patch_grid^2 патчей × (4 статистики + n_orient_bins) + 3 глобальных
    assert v.shape[0] == 4 * 4 * (4 + 9) + 3


def test_scharr_deterministic():
    e = ScharrExtractor(img_size=64, verbose=False)
    rng = np.random.default_rng(RNG_SEED)
    img = _make_gray(rng, fake=True)
    np.testing.assert_array_equal(e._extract(img), e._extract(img))


def test_scharr_separates_real_fake():
    e = ScharrExtractor(img_size=64, patch_grid=2, verbose=False)
    assert _separates(e)


def test_scharr_threshold_zeroes_weak_magnitudes():
    rng = np.random.default_rng(RNG_SEED)
    img = _make_gray(rng, fake=False)
    lo = ScharrExtractor(img_size=64, threshold=None, verbose=False)._extract(img)
    hi = ScharrExtractor(img_size=64, threshold=1e6, verbose=False)._extract(img)
    # огромный порог убивает всю магнитуду: глобальный mean (позиция -3) -> 0
    assert hi[-3] == 0.0
    assert lo[-3] > 0.0


# ---------- LoG fixes ----------


def test_log_zero_crossings_constant_image_is_empty():
    img = np.full((32, 32), 100, dtype=np.uint8)
    zc = _log_zero_crossings(img, sigma=1.2, zc_threshold=0.5)
    # нули LoG считаются положительными -> нет ложных смен знака на плоском поле
    assert zc.sum() == 0.0


def test_log_threshold_suppresses_noise_crossings():
    rng = np.random.default_rng(RNG_SEED)
    img = _make_gray(rng, fake=False)
    zc_no_thr = _log_zero_crossings(img, sigma=1.2, zc_threshold=0.0)
    zc_thr = _log_zero_crossings(img, sigma=1.2, zc_threshold=1.5)
    assert zc_thr.sum() < zc_no_thr.sum()


def test_log_extractor_new_defaults():
    e = LoGExtractor(verbose=False)
    assert e.sigma == 1.2
    assert e.zc_threshold == 0.5


def test_log_extractor_shape_and_determinism():
    e = LoGExtractor(img_size=64, patch_size=16, verbose=False)
    rng = np.random.default_rng(RNG_SEED)
    img = _make_gray(rng, fake=True)
    v1, v2 = e._extract(img), e._extract(img)
    np.testing.assert_array_equal(v1, v2)
    assert v1.shape[0] == (64 // 16) ** 2 * 4 + 3


def test_log_extractor_separates_real_fake():
    e = LoGExtractor(img_size=64, patch_size=16, verbose=False)
    assert _separates(e)


# ---------- Canny (L2gradient) ----------


def test_canny_shape_and_determinism():
    e = CannyEdgeExtractor(img_size=64, patch_grid=2, verbose=False)
    rng = np.random.default_rng(RNG_SEED)
    img = _make_gray(rng, fake=True)
    v1, v2 = e._extract(img), e._extract(img)
    np.testing.assert_array_equal(v1, v2)
    assert v1.shape[0] == 2 * 2 * (2 + 9) + 2
    assert v1.dtype == np.float32


# ---------- color.py: диапазоны гистограмм ----------


def test_channel_ranges():
    assert _channel_ranges("HSV") == [(0, 180), (0, 256), (0, 256)]
    assert _channel_ranges("RGB") == [(0, 256)] * 3
    assert _channel_ranges("YCbCr") == [(0, 256)] * 3


def test_color_histogram_sums_to_one_per_channel():
    e = ColorHistogramExtractor(img_size=64, n_bins=16, colorspace="HSV", verbose=False)
    rng = np.random.default_rng(RNG_SEED)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    v = e._extract(img)
    assert v.shape[0] == 3 * 16
    # нормировка на сумму: каждая канальная гистограмма суммируется в 1
    for ch in range(3):
        assert v[ch * 16 : (ch + 1) * 16].sum() == pytest.approx(1.0, abs=1e-5)


def test_color_histogram_h_range_covers_full_hue():
    # насыщенный градиент по hue: H заполняет 0..180; с range (0,256)
    # верхние бины (H > 180) были бы пусты всегда
    hsv = np.zeros((64, 64, 3), dtype=np.uint8)
    hsv[..., 0] = np.linspace(0, 179, 64).astype(np.uint8)[None, :]
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    import cv2

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    e = ColorHistogramExtractor(img_size=64, n_bins=16, colorspace="HSV", verbose=False)
    v = e._extract(rgb)
    h_hist = v[:16]
    # при корректном range (0, 180) заполнены и верхние бины гистограммы H
    assert h_hist[-2:].sum() > 0.0


# ---------- color.py: циркулярные статистики H ----------


def test_circular_mean_wraps_around_zero():
    h_179 = np.full(100, 179.0)
    h_1 = np.full(100, 1.0)
    cm179, _ = _circular_mean_std_hue(h_179)
    cm1, _ = _circular_mean_std_hue(h_1)
    # циркулярное расстояние между средними мало (оба у границы 0/180)
    d = min(abs(cm179 - cm1), 180.0 - abs(cm179 - cm1))
    assert d < 5.0
    # обычный mean дал бы разницу ~178
    assert abs(h_179.mean() - h_1.mean()) > 170.0


def test_circular_mean_mixed_boundary_not_ninety():
    h = np.concatenate([np.full(50, 179.0), np.full(50, 1.0)])
    cm, cs = _circular_mean_std_hue(h)
    # обычный mean = 90 (центр шкалы); циркулярный — у границы 0/180
    assert h.mean() == pytest.approx(90.0)
    d0 = min(cm, 180.0 - cm)
    assert d0 < 5.0
    # разброс между двумя соседними значениями мал
    assert cs < 0.2


def test_color_moments_hsv_circular_vs_regular():
    """У HSV-изображения с H около границы 0/180 циркулярный mean не равен ~90."""
    import cv2

    rng = np.random.default_rng(RNG_SEED)
    hsv = np.zeros((32, 32, 3), dtype=np.uint8)
    h_vals = rng.choice([0, 1, 178, 179], size=(32, 32))
    hsv[..., 0] = h_vals
    hsv[..., 1] = 255
    hsv[..., 2] = 200
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    e = ColorMomentsExtractor(img_size=32, patch_sizes=(32,), colorspace="HSV", verbose=False)
    v = e._extract(rgb)
    mean_h = v[0]  # первый признак первого канала (H) — циркулярный mean
    d0 = min(mean_h, 180.0 - mean_h)
    assert d0 < 10.0


def test_color_moments_hsv_shape_and_determinism():
    e = ColorMomentsExtractor(img_size=64, patch_sizes=(32,), colorspace="HSV", verbose=False)
    rng = np.random.default_rng(RNG_SEED)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    v1, v2 = e._extract(img), e._extract(img)
    np.testing.assert_array_equal(v1, v2)
    # 4 патча × 3 канала × 9 статистик
    assert v1.shape[0] == 4 * 3 * 9


# ---------- color.py: порядок каналов YCbCr ----------


def test_ycbcr_channel_order_semantics():
    blue = np.zeros((8, 8, 3), dtype=np.uint8)
    blue[..., 2] = 255  # чистый синий (RGB)
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255  # чистый красный (RGB)
    y_b = _convert_colorspace(blue, "YCbCr")
    y_r = _convert_colorspace(red, "YCbCr")
    # после переупорядочивания: канал 1 = Cb (макс. для синего), канал 2 = Cr (для красного)
    assert y_b[..., 1].mean() > y_b[..., 2].mean()
    assert y_r[..., 2].mean() > y_r[..., 1].mean()


def test_ycbcr_chi_square_extractor_shape_and_determinism():
    e = YCbCrChiSquareExtractor(img_size=64, n_bins=16, patch_grid=2, verbose=False)
    rng = np.random.default_rng(RNG_SEED)
    img = _make_rgb(rng, fake=True)
    v1, v2 = e._extract(img), e._extract(img)
    np.testing.assert_array_equal(v1, v2)
    # 2 канала × (9 глобальных + 4 патча × (9 моментов + 1 chi-square))
    assert v1.shape[0] == 2 * (9 + 4 * 10)
    assert np.isfinite(v1).all()


# ---------- пайплайн GRAD-2 ----------


def test_grad2_pipeline_from_cfg_and_fit(tiny_splits):
    from deepfakes.models.classical import ModelSpec
    from deepfakes.pipelines.grad2 import Grad2Pipeline

    cfg = {"img_size": 64, "patch_grid": 2, "n_orient_bins": 9, "threshold": None}
    pipe = Grad2Pipeline.from_cfg(cfg, ModelSpec(name="logreg", params={}), cache_dir=None)
    assert pipe.pipeline_id == "GRAD-2"
    assert pipe.group == "grad"
    assert isinstance(pipe.extractor, ScharrExtractor)
    assert pipe.extractor.img_size == 64
    artifacts = pipe.fit(
        tiny_splits["train"]["X"],
        tiny_splits["train"]["y"],
        X_val_paths=tiny_splits["val"]["X"],
        y_val=tiny_splits["val"]["y"],
    )
    assert artifacts.feature_dim == 2 * 2 * (4 + 9) + 3
    probs = pipe.predict_proba(tiny_splits["test"]["X"])
    assert probs.shape == (len(tiny_splits["test"]["X"]),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
