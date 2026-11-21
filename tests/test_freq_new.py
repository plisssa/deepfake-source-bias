"""Тесты частотного блока: мат.фиксы FREQ-4, новые FREQ-1 (FullFFT) и FREQ-5 (CWT)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deepfakes.features.freq import (
    CWTScalogramExtractor,
    FullFFTExtractor,
    HFEnergyExtractor,
    WaveletExtractor,
)

SIZE = 64


def _noise_img(seed: int = 0, size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(128, 30, (size, size)).clip(0, 255).astype(np.uint8)


def _smooth_img(seed: int = 0, size: int = SIZE) -> np.ndarray:
    """1/f-подобное изображение: сильно сглаженный шум (HF подавлены)."""
    img = _noise_img(seed, size).astype(np.float32)
    return cv2.GaussianBlur(img, (0, 0), 3.0).clip(0, 255).astype(np.uint8)


def _fake_img(seed: int = 0, size: int = SIZE) -> np.ndarray:
    """Синтетический fake: шум + шахматка с клеткой 2px (HF GAN-артефакт)."""
    rng = np.random.default_rng(seed)
    ii, jj = np.indices((size, size))
    chess = (((ii // 2) + (jj // 2)) % 2).astype(np.float32) * 12.0
    return (rng.normal(128, 30, (size, size)) + chess).clip(0, 255).astype(np.uint8)


def _separates(feats_real: np.ndarray, feats_fake: np.ndarray) -> bool:
    """Хотя бы один признак различает классы: разрыв средних > сумма std."""
    mr, mf = feats_real.mean(axis=0), feats_fake.mean(axis=0)
    sr, sf = feats_real.std(axis=0), feats_fake.std(axis=0)
    return bool(np.any(np.abs(mr - mf) > (sr + sf + 1e-12)))


# ---------- HFEnergyExtractor: мат.фиксы ----------


def test_hfe_shape_and_determinism():
    e = HFEnergyExtractor(img_size=SIZE, n_bins=6, verbose=False)
    img = _noise_img(1)
    f1 = e._extract(img)
    f2 = e._extract(img)
    assert f1.shape == (6 + 1 + 1 + 2,)
    assert f1.dtype == np.float32
    np.testing.assert_array_equal(f1, f2)


def test_hfe_slope_white_noise_near_zero():
    """У белого шума плоский спектр — азимутальный slope около нуля."""
    e = HFEnergyExtractor(img_size=SIZE, n_bins=6, verbose=False)
    slopes = [e._extract(_noise_img(s))[6] for s in range(5)]
    assert all(abs(s) < 0.6 for s in slopes), slopes


def test_hfe_slope_smooth_more_negative():
    """У сглаженного (1/f-подобного) изображения slope заметно отрицательнее."""
    e = HFEnergyExtractor(img_size=SIZE, n_bins=6, verbose=False)
    slope_noise = float(np.mean([e._extract(_noise_img(s))[6] for s in range(5)]))
    slope_smooth = float(np.mean([e._extract(_smooth_img(s))[6] for s in range(5)]))
    assert slope_smooth < slope_noise - 1.0
    assert slope_smooth < -1.0


def test_hfe_flatness_in_unit_interval():
    """flatness = gm/am по мощности без DC — строго в (0, 1]."""
    e = HFEnergyExtractor(img_size=SIZE, n_bins=6, verbose=False)
    for img in (_noise_img(2), _smooth_img(2), _fake_img(2)):
        flat = e._extract(img)[7]
        assert 0.0 < flat <= 1.0


def test_hfe_normalized_rings_sum_to_one():
    """normalize=True: доли энергии по кольцам покрывают весь спектр (сумма ~1)."""
    e = HFEnergyExtractor(
        img_size=SIZE,
        n_bins=8,
        include_slope=False,
        include_flatness=False,
        include_basic_stats=False,
        verbose=False,
    )
    rings = e._extract(_noise_img(3))
    assert rings.shape == (8,)
    assert abs(float(rings.sum()) - 1.0) < 1e-4
    # последний бин (углы спектра) больше не пустой
    assert rings[-1] > 0.0


def test_hfe_window_param():
    e_hann = HFEnergyExtractor(img_size=SIZE, verbose=False)  # default hann
    e_none = HFEnergyExtractor(img_size=SIZE, window="none", verbose=False)
    img = _noise_img(4)
    assert not np.array_equal(e_hann._extract(img), e_none._extract(img))
    e_bad = HFEnergyExtractor(img_size=SIZE, window="hamming", verbose=False)
    with pytest.raises(ValueError):
        e_bad._extract(img)


def test_hfe_separates_real_fake():
    e = HFEnergyExtractor(img_size=SIZE, verbose=False)
    fr = np.vstack([e._extract(_noise_img(s)) for s in range(10)])
    ff = np.vstack([e._extract(_fake_img(s)) for s in range(10)])
    assert _separates(fr, ff)


# ---------- FullFFTExtractor (FREQ-1) ----------


def test_fullfft_shape_and_determinism():
    e = FullFFTExtractor(img_size=SIZE, verbose=False)
    img = _noise_img(5)
    f1 = e._extract(img)
    f2 = e._extract(img)
    assert f1.shape == (16 + 8 + 2 + 16,)
    assert f1.dtype == np.float32
    np.testing.assert_array_equal(f1, f2)


def test_fullfft_shape_without_phase():
    e = FullFFTExtractor(
        img_size=SIZE, radial_bins=10, angular_bins=4, include_phase=False, verbose=False
    )
    assert e._extract(_noise_img(5)).shape == (10 + 4 + 2,)


def test_fullfft_angular_fractions_and_flatness():
    e = FullFFTExtractor(img_size=SIZE, verbose=False)
    f = e._extract(_noise_img(6))
    angular = f[16:24]
    assert np.all(angular >= 0.0)
    assert abs(float(angular.sum()) - 1.0) < 1e-4
    flatness = f[25]
    assert 0.0 < flatness <= 1.0
    phase = f[26:42]
    assert np.all(phase >= 0.0) and np.all(phase <= 1.0 + 1e-6)


def test_fullfft_separates_real_fake():
    e = FullFFTExtractor(img_size=SIZE, verbose=False)
    fr = np.vstack([e._extract(_noise_img(s)) for s in range(10)])
    ff = np.vstack([e._extract(_fake_img(s)) for s in range(10)])
    assert _separates(fr, ff)


# ---------- CWTScalogramExtractor (FREQ-5) ----------


@pytest.mark.parametrize("wavelet", ["morl", "mexh"])
def test_cwt_shape_and_determinism(wavelet):
    e = CWTScalogramExtractor(img_size=SIZE, wavelet=wavelet, verbose=False)
    img = _noise_img(7)
    f1 = e._extract(img)
    f2 = e._extract(img)
    assert f1.shape == (2 * 4 * 3,)  # 2 сигнала x 4 scale x 3 статистики
    assert f1.dtype == np.float32
    np.testing.assert_array_equal(f1, f2)


def test_cwt_separates_real_banded_fake():
    """Диагональная шахматка сбалансирована по строкам/столбцам и в проекциях
    сокращается (ограничение задокументировано в докстринге экстрактора),
    поэтому fake для CWT — шум + периодический banding (период 8px)."""
    e = CWTScalogramExtractor(img_size=SIZE, verbose=False)

    def _banded(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ii = np.indices((SIZE, SIZE))[0]
        band = np.sin(2 * np.pi * ii / 8.0).astype(np.float32) * 10.0
        return (rng.normal(128, 30, (SIZE, SIZE)) + band).clip(0, 255).astype(np.uint8)

    fr = np.vstack([e._extract(_noise_img(s)) for s in range(10)])
    ff = np.vstack([e._extract(_banded(s)) for s in range(10)])
    assert _separates(fr, ff)


# ---------- WaveletExtractor (FREQ-3): без fallback ----------


def test_wavelet_fixed_dim_no_fallback():
    """5 статистик x (LL + 3 detail-subband x level) — размерность не «плавает»."""
    e = WaveletExtractor(img_size=SIZE, wavelet="db2", level=3, verbose=False)
    f = e._extract(_noise_img(8))
    assert f.shape == (5 * (1 + 3 * 3),)
    assert not hasattr(e, "_laplacian_fallback")
    assert not hasattr(e, "_has_pywt")


# ---------- пайплайны freq1 / freq5 ----------


def test_freq1_freq5_pipelines_from_cfg():
    from deepfakes.models.classical import ModelSpec
    from deepfakes.pipelines.freq1 import Freq1Pipeline
    from deepfakes.pipelines.freq5 import Freq5Pipeline

    spec = ModelSpec(name="logreg", params={})
    p1 = Freq1Pipeline.from_cfg({"img_size": SIZE}, spec)
    p5 = Freq5Pipeline.from_cfg({"img_size": SIZE, "wavelet": "mexh"}, spec)
    assert p1.pipeline_id == "FREQ-1" and p1.group == "freq"
    assert p5.pipeline_id == "FREQ-5" and p5.group == "freq"
    assert isinstance(p1.extractor, FullFFTExtractor)
    assert isinstance(p5.extractor, CWTScalogramExtractor)
    assert p5.extractor.wavelet == "mexh"
