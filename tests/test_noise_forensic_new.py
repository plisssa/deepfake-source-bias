"""Тесты блока шум+форензика: center_crop в NOISE-2/NOISE-3, фикс дублей, ELA (FOR-1)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deepfakes.features.forensic import ELAExtractor
from deepfakes.features.noise import JPEGBlockinessExtractor, PRNUExtractor

RNG = np.random.default_rng(42)


@pytest.fixture(autouse=True)
def _reset_rng():
    """Одинаковые входные данные в каждом тесте независимо от порядка запуска."""
    global RNG
    RNG = np.random.default_rng(42)


def _noise_gray(size: int = 64) -> np.ndarray:
    return RNG.normal(128, 20, (size, size)).clip(0, 255).astype(np.uint8)


def _noise_rgb(size: int = 64, noise_std: float = 20.0) -> np.ndarray:
    return RNG.normal(128, noise_std, (size, size, 3)).clip(0, 255).astype(np.uint8)


def _chess2(size: int, amp: float = 12.0) -> np.ndarray:
    """Шахматка с клеткой 2×2 px (высокочастотный «GAN-артефакт»)."""
    yy, xx = np.mgrid[0:size, 0:size]
    return (((yy // 2 + xx // 2) % 2) * amp).astype(np.float32)


def _fake_gray(size: int = 64) -> np.ndarray:
    base = RNG.normal(128, 20, (size, size)) + _chess2(size)
    return base.clip(0, 255).astype(np.uint8)


def _fake_rgb(size: int = 64, noise_std: float = 20.0) -> np.ndarray:
    base = RNG.normal(128, noise_std, (size, size, 3)) + _chess2(size)[:, :, None]
    return base.clip(0, 255).astype(np.uint8)


# ---------- center_crop ----------


def test_center_crop_no_interpolation(tmp_path):
    """Кроп реально без интерполяции: центральное окно совпадает попиксельно."""
    big = RNG.integers(0, 256, (400, 400), dtype=np.uint8)
    path = str(tmp_path / "big.png")
    assert cv2.imwrite(path, big)
    for ext in (
        JPEGBlockinessExtractor(img_size=256),
        PRNUExtractor(img_size=256),
    ):
        assert ext.resize_mode == "center_crop"  # default
        img = ext._read_image(path)
        assert img.shape == (256, 256)
        y0 = (400 - 256) // 2
        np.testing.assert_array_equal(img, big[y0 : y0 + 256, y0 : y0 + 256])


def test_center_crop_pads_small_images(tmp_path):
    small = RNG.integers(0, 256, (100, 120), dtype=np.uint8)
    path = str(tmp_path / "small.png")
    assert cv2.imwrite(path, small)
    ext = JPEGBlockinessExtractor(img_size=256)
    img = ext._read_image(path)
    assert img.shape == (256, 256)


def test_resize_mode_resize_keeps_old_behavior(tmp_path):
    big = RNG.integers(0, 256, (400, 400), dtype=np.uint8)
    path = str(tmp_path / "big2.png")
    assert cv2.imwrite(path, big)
    ext = JPEGBlockinessExtractor(img_size=256, resize_mode="resize")
    img = ext._read_image(path)
    assert img.shape == (256, 256)
    expected = cv2.resize(big, (256, 256), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(img, expected)


# ---------- формы векторов ----------


def test_blockiness_patch_grid_1_no_duplicates():
    """При patch_grid==1 глобальные статистики считаются один раз: ровно 6 чисел."""
    ext = JPEGBlockinessExtractor(img_size=64, patch_grid=1)
    feats = ext._extract(_noise_gray(64))
    assert feats.shape == (6,)


def test_blockiness_patch_grid_2_shape():
    ext = JPEGBlockinessExtractor(img_size=64, patch_grid=2)
    feats = ext._extract(_noise_gray(64))
    assert feats.shape == ((4 + 1) * 6,)  # 4 патча + глобально


@pytest.mark.parametrize("grid,expected", [(1, 12 + 6), (4, 12 + 16 * 6)])
def test_ela_shape(grid, expected):
    ext = ELAExtractor(img_size=64, patch_grid=grid)
    feats = ext._extract(_noise_rgb(64))
    assert feats.shape == (expected,)
    assert feats.dtype == np.float32


# ---------- детерминизм ----------


@pytest.mark.parametrize(
    "ext,img",
    [
        (JPEGBlockinessExtractor(img_size=64, patch_grid=2), _noise_gray(64)),
        (PRNUExtractor(img_size=64, patch_grid=2), _noise_gray(64)),
        (ELAExtractor(img_size=64, patch_grid=2), _noise_rgb(64)),
    ],
)
def test_deterministic(ext, img):
    f1 = ext._extract(img.copy())
    f2 = ext._extract(img.copy())
    np.testing.assert_array_equal(f1, f2)


# ---------- ELA: пережатый vs непережатый ----------


def test_ela_distinguishes_recompressed():
    """Уже сжатое JPEG-изображение даёт меньшую ошибку повторного сжатия."""
    ext = ELAExtractor(img_size=64, quality=90, patch_grid=1)
    raw = _noise_rgb(64)  # никогда не проходило JPEG
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    assert ok
    jpeg = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    f_raw = ext._extract(raw)
    f_jpeg = ext._extract(jpeg)
    # feats[0], feats[4], feats[8] — глобальные mean по каналам
    mean_raw = f_raw[[0, 4, 8]].sum()
    mean_jpeg = f_jpeg[[0, 4, 8]].sum()
    assert mean_jpeg < mean_raw


# ---------- real vs fake (шум vs шум+шахматка 2px) ----------


def _separates(feats_real: np.ndarray, feats_fake: np.ndarray) -> bool:
    """Есть ли хоть один признак, где средние разъезжаются сильнее разброса."""
    mu_r, mu_f = feats_real.mean(axis=0), feats_fake.mean(axis=0)
    sd = feats_real.std(axis=0) + feats_fake.std(axis=0) + 1e-8
    return bool((np.abs(mu_r - mu_f) > sd).any())


def test_blockiness_separates_real_fake():
    ext = JPEGBlockinessExtractor(img_size=64, patch_grid=1)
    fr = np.stack([ext._extract(_noise_gray(64)) for _ in range(8)])
    ff = np.stack([ext._extract(_fake_gray(64)) for _ in range(8)])
    assert _separates(fr, ff)


def test_ela_separates_real_fake():
    # Шум послабее (std=5), чтобы шахматка была ведущей ВЧ-компонентой: ELA
    # ловит именно доп. энергию высоких частот, а не сам шум; различимость
    # не обязана быть идеальной при шуме, сравнимом с паттерном.
    ext = ELAExtractor(img_size=64, quality=90, patch_grid=1)
    fr = np.stack([ext._extract(_noise_rgb(64, noise_std=5.0)) for _ in range(8)])
    ff = np.stack([ext._extract(_fake_rgb(64, noise_std=5.0)) for _ in range(8)])
    assert _separates(fr, ff)


def test_prnu_separates_real_fake():
    ext = PRNUExtractor(img_size=64, patch_grid=1)
    fr = np.stack([ext._extract(_noise_gray(64)) for _ in range(8)])
    ff = np.stack([ext._extract(_fake_gray(64)) for _ in range(8)])
    assert _separates(fr, ff)


# ---------- пайплайн FOR-1 smoke ----------


def test_ela_pipeline_smoke(tiny_splits):
    from deepfakes.models.classical import ModelSpec
    from deepfakes.pipelines.ela import ElaPipeline

    cfg = {"img_size": 64, "quality": 90, "patch_grid": 2}
    pipe = ElaPipeline.from_cfg(cfg, ModelSpec(name="logreg", params={}), cache_dir=None)
    artifacts = pipe.fit(
        tiny_splits["train"]["X"],
        tiny_splits["train"]["y"],
        X_val_paths=tiny_splits["val"]["X"],
        y_val=tiny_splits["val"]["y"],
    )
    assert artifacts.feature_dim == 12 + 4 * 6
    probs = pipe.predict_proba(tiny_splits["test"]["X"])
    assert probs.shape == (len(tiny_splits["test"]["X"]),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
