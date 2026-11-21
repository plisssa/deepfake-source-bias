"""Тесты новых текстурных экстракторов: LTP (TEX-3), LPQ, банк Габора (TEX-5).

Проверяются: формы векторов, детерминизм и то, что признаки различают
синтетические real (шум) и fake (шум + шахматка периода 2px) хотя бы по
одному признаку.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from deepfakes.features.texture import GaborBankExtractor, LPQExtractor, LTPExtractor


@pytest.fixture(scope="module")
def texture_dataset(tmp_path_factory) -> dict:
    """Маленький датасет: real = шум, fake = шум + шахматка 2px."""
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("texture_ds")
    n_per = 10
    real_paths, fake_paths = [], []
    chess = np.zeros((64, 64), dtype=np.float32)
    chess[::2, ::2] = 14
    chess[1::2, 1::2] = 14
    for i in range(n_per):
        real = rng.normal(128, 20, (64, 64)).clip(0, 255).astype(np.uint8)
        p_real = root / f"real_{i}.png"
        Image.fromarray(real).save(p_real)
        real_paths.append(str(p_real))

        fake = (rng.normal(128, 20, (64, 64)) + chess).clip(0, 255).astype(np.uint8)
        p_fake = root / f"fake_{i}.png"
        Image.fromarray(fake).save(p_fake)
        fake_paths.append(str(p_fake))
    return {"real": np.array(real_paths), "fake": np.array(fake_paths)}


def _extractors() -> list:
    """Экстракторы с ожидаемой размерностью признаков (img_size=64 для скорости)."""
    return [
        (LTPExtractor(img_size=64, radii=(1, 2), t=5.0, verbose=False), 2 * 2 * 256),
        (LPQExtractor(img_size=64, win_size=7, verbose=False), 256),
        (GaborBankExtractor(img_size=64, verbose=False), 4 * 3 * 3),
    ]


@pytest.mark.parametrize("extractor,expected_dim", _extractors(), ids=lambda p: type(p).__name__)
def test_output_shape(texture_dataset, extractor, expected_dim):
    paths = texture_dataset["real"][:4]
    X = extractor.transform(paths)
    assert X.shape == (len(paths), expected_dim)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


@pytest.mark.parametrize("extractor,_dim", _extractors(), ids=lambda p: type(p).__name__)
def test_determinism(texture_dataset, extractor, _dim):
    paths = texture_dataset["real"][:3]
    X1 = extractor.transform(paths)
    X2 = extractor.transform(paths)
    np.testing.assert_array_equal(X1, X2)


@pytest.mark.parametrize("extractor,_dim", _extractors(), ids=lambda p: type(p).__name__)
def test_discriminates_real_vs_fake(texture_dataset, extractor, _dim):
    """Хотя бы один признак разделяет классы (effect size > 0.5)."""
    X_real = extractor.transform(texture_dataset["real"])
    X_fake = extractor.transform(texture_dataset["fake"])
    mu_r, mu_f = X_real.mean(axis=0), X_fake.mean(axis=0)
    sd = 0.5 * (X_real.std(axis=0) + X_fake.std(axis=0)) + 1e-12
    effect = np.abs(mu_r - mu_f) / sd
    assert effect.max() > 0.5, f"max effect size = {effect.max():.3f}"


def test_ltp_upper_lower_split():
    """Ровный градиент: сосед справа всегда больше центра на 10 > t=5 —
    верхний код имеет бит 3 (сосед (0, +1)), нижний — бит 7 (сосед (0, -1))."""
    ramp = np.tile(np.arange(0, 640, 10, dtype=np.float32), (64, 1))
    ext = LTPExtractor(img_size=64, radii=(1,), t=5.0, verbose=False)
    upper, lower = ext._ltp_codes(ramp, radius=1)
    # Во внутренней области (без границ-отражений) биты известны точно
    inner_u = upper[1:-1, 1:-1]
    inner_l = lower[1:-1, 1:-1]
    assert (inner_u & (1 << 3)).all()  # правый сосед >= центр + t
    assert (inner_l & (1 << 7)).all()  # левый сосед <= центр - t
    assert not (inner_u & (1 << 7)).any()
    assert not (inner_l & (1 << 3)).any()


def test_ltp_zero_on_flat():
    """Плоское изображение: все s = 0, оба кода нулевые, вся масса в бине 0."""
    flat = np.full((32, 32), 100, dtype=np.uint8)
    ext = LTPExtractor(img_size=32, radii=(1,), t=5.0, verbose=False)
    feats = ext._extract(flat)
    # гистограммы density: бин 0 верхнего и нижнего кодов содержит всю массу
    assert feats[0] > 0
    assert feats[1:256].sum() == pytest.approx(0.0)
    assert feats[256] > 0
    assert feats[257:512].sum() == pytest.approx(0.0)


def test_lpq_histogram_is_density():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    ext = LPQExtractor(img_size=64, verbose=False)
    feats = ext._extract(img)
    assert feats.shape == (256,)
    assert feats.sum() == pytest.approx(1.0, rel=1e-5)


def test_gabor_patch_grid():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    ext = GaborBankExtractor(img_size=64, patch_grid=2, verbose=False)
    feats = ext._extract(img)
    assert feats.shape == (4 * 3 * 3 * 4,)  # 36 признаков на каждый из 4 патчей
