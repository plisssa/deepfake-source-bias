"""Тесты блока аугментаций: реестр AUGMENTERS + универсальный хук BasePipeline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from deepfakes.augment import AUGMENTERS
from deepfakes.pipelines.base import BasePipeline

AUG_PARAMS = {
    "gaussian_noise": {"sigma": 8},
    "poisson_noise": {},
    "gaussian_poisson": {"sigma": 5},
    "fft_amplitude_jitter": {"strength": 0.2},
    "hflip": {},
    "random_crop_resize": {"scale": 0.8},
}


def _make_img(rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """Асимметричная картинка: шум + градиент (чтобы hflip её менял)."""
    noise = rng.normal(128, 30, (size, size, 3))
    gradient = np.linspace(0, 60, size)[None, :, None]
    return np.clip(noise + gradient, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", sorted(AUGMENTERS))
def test_augmenter_deterministic_and_changes_image(name):
    img = _make_img(np.random.default_rng(0))
    params = AUG_PARAMS[name]
    a = AUGMENTERS[name](img, np.random.default_rng(7), **params)
    b = AUGMENTERS[name](img, np.random.default_rng(7), **params)
    # детерминизм при том же seed
    assert np.array_equal(a, b), f"{name}: не детерминирован при одинаковом rng"
    # форма и тип сохраняются
    assert a.shape == img.shape
    assert a.dtype == np.uint8
    # изображение реально меняется
    assert not np.array_equal(a, img), f"{name}: не изменил изображение"


def test_augmenters_grayscale_supported():
    """Все аугментеры работают и с 2D grayscale-входом."""
    img = _make_img(np.random.default_rng(1))[:, :, 0]
    for name, fn in AUGMENTERS.items():
        out = fn(img, np.random.default_rng(3), **AUG_PARAMS[name])
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name


@pytest.fixture()
def small_dataset(tmp_path):
    """10 синтетических картинок: 5 real (шум) + 5 fake (шум + шахматка 2px)."""
    rng = np.random.default_rng(42)
    paths, labels = [], []
    chess = np.indices((64, 64)).sum(axis=0) // 2 % 2 * 30.0
    for i in range(10):
        img = rng.normal(128, 20, (64, 64, 3))
        label = i % 2
        if label == 1:
            img = img + chess[:, :, None]
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(p)
        paths.append(str(p))
        labels.append(label)
    return np.array(paths, dtype=object), np.array(labels)


def _make_pipeline(cache_dir, augment_cfg):
    # augment_paths не использует extractor/model — для unit-теста хука
    # достаточно "пустого" пайплайна.
    return BasePipeline(
        extractor=None, model_spec=None, cache_dir=str(cache_dir), augment_cfg=augment_cfg
    )


def test_hook_without_cfg_is_noop(tmp_path, small_dataset):
    X, y = small_dataset
    pipe = _make_pipeline(tmp_path / "cache", augment_cfg=None)
    X2, y2 = pipe.augment_paths(X, y)
    assert X2 is X and y2 is y


def test_hook_augments_and_reuses_files(tmp_path, small_dataset):
    X, y = small_dataset
    cache_dir = tmp_path / "cache"
    cfg = {"name": "gaussian_noise", "times": 1, "params": {"sigma": 8}, "seed": 1}
    pipe = _make_pipeline(cache_dir, augment_cfg=cfg)

    X_aug, y_aug = pipe.augment_paths(X, y)
    assert len(X_aug) == 20 and len(y_aug) == 20
    # метки сохранены: первые 10 исходные, вторые 10 повторяют их
    assert np.array_equal(y_aug[:10], y) and np.array_equal(y_aug[10:], y)

    aug_dirs = list(cache_dir.glob("aug_gaussian_noise_*"))
    assert len(aug_dirs) == 1
    n_files = len(list(aug_dirs[0].iterdir()))
    assert n_files == 10

    # повторный вызов переиспользует файлы (директория не растёт) и детерминирован
    X_aug2, y_aug2 = pipe.augment_paths(X, y)
    assert list(X_aug2) == list(X_aug)
    assert np.array_equal(y_aug2, y_aug)
    assert len(list(aug_dirs[0].iterdir())) == n_files

    # аугментированные картинки отличаются от исходных
    orig = np.asarray(Image.open(str(X[0])))
    aug = np.asarray(Image.open(str(X_aug[10])))
    assert not np.array_equal(orig, aug)


def test_hook_times_2(tmp_path, small_dataset):
    X, y = small_dataset
    cfg = {"name": "gaussian_noise", "times": 2, "params": {"sigma": 8}, "seed": 1}
    pipe = _make_pipeline(tmp_path / "cache", augment_cfg=cfg)
    X_aug, y_aug = pipe.augment_paths(X, y)
    assert len(X_aug) == 30 and len(y_aug) == 30
    # копии k=0 и k=1 одного файла различаются (разные rng-подпотоки)
    a = np.asarray(Image.open(str(X_aug[10])))
    b = np.asarray(Image.open(str(X_aug[11])))
    assert not np.array_equal(a, b)


def test_augmentation_preserves_real_fake_separation(small_dataset, tmp_path):
    """Признак (HF-энергия шахматки 2px) различает классы и после аугментации."""
    X, y = small_dataset
    cfg = {"name": "gaussian_noise", "times": 1, "params": {"sigma": 8}, "seed": 1}
    pipe = _make_pipeline(tmp_path / "cache", augment_cfg=cfg)
    X_aug, y_aug = pipe.augment_paths(X, y)

    def hf_energy(path: str) -> float:
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        fft = np.fft.fftshift(np.fft.fft2(img))
        h, w = img.shape
        r = h // 4
        mask = np.ones_like(img, dtype=bool)
        mask[h // 2 - r : h // 2 + r, w // 2 - r : w // 2 + r] = False
        return float(np.abs(fft)[mask].mean())

    feats = np.array([hf_energy(str(p)) for p in X_aug])
    assert feats[y_aug == 1].mean() > feats[y_aug == 0].mean()
