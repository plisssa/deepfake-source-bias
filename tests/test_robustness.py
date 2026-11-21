"""Стенд устойчивости: деградации корректны и детерминированы."""

from __future__ import annotations

import numpy as np
import pytest

from deepfakes.augment.degrade import DEGRADATIONS


@pytest.fixture()
def img() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_shape_dtype_preserved(name, img):
    fn, params = DEGRADATIONS[name]
    out = fn(img, np.random.default_rng(0), **params)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", [n for n in DEGRADATIONS if n != "clean"])
def test_degradation_changes_image(name, img):
    fn, params = DEGRADATIONS[name]
    out = fn(img, np.random.default_rng(0), **params)
    assert not np.array_equal(out, img), f"{name} не изменила изображение"


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_deterministic_with_same_seed(name, img):
    fn, params = DEGRADATIONS[name]
    a = fn(img, np.random.default_rng(7), **params)
    b = fn(img, np.random.default_rng(7), **params)
    np.testing.assert_array_equal(a, b)


def test_clean_is_identity(img):
    fn, params = DEGRADATIONS["clean"]
    np.testing.assert_array_equal(fn(img, np.random.default_rng(0), **params), img)


def test_stronger_jpeg_degrades_more(img):
    """q=50 должно отклонять изображение сильнее, чем q=90."""
    fn, _ = DEGRADATIONS["jpeg90"]
    d90 = np.abs(fn(img, None, quality=90).astype(int) - img.astype(int)).mean()
    d50 = np.abs(fn(img, None, quality=50).astype(int) - img.astype(int)).mean()
    assert d50 > d90


def test_materialize_keeps_labels_aligned(tmp_path):
    """Нечитаемый файл должен выпасть ВМЕСТЕ со своей меткой, а не сдвинуть их.

    Регрессия на срез `y[:len(X)]`: он оставлял метку выпавшего файла и сдвигал
    все последующие на единицу — молча, с правдоподобным AUC."""
    import cv2

    from scripts.run_robustness import _materialize

    rng = np.random.default_rng(11)
    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        paths.append(str(p))
    bad = tmp_path / "img_broken.png"  # не изображение
    bad.write_bytes(b"not an image")
    paths.insert(2, str(bad))  # битый файл в середине — самый опасный случай

    labels = np.array([0, 1, 1, 0, 1], dtype=np.int8)
    X, kept = _materialize(np.array(paths, dtype=object), "jpeg75", tmp_path / "out", seed=1)

    assert len(X) == len(kept) == 4, "битый файл должен выпасть"
    assert list(kept) == [0, 1, 3, 4], "индексы должны указывать на исходные позиции"
    np.testing.assert_array_equal(labels[kept], np.array([0, 1, 0, 1], dtype=np.int8))


def test_materialize_clean_returns_all_indices():
    from scripts.run_robustness import _materialize

    paths = np.array(["a.png", "b.png", "c.png"], dtype=object)
    X, kept = _materialize(paths, "clean", None, seed=0)
    np.testing.assert_array_equal(X, paths)
    np.testing.assert_array_equal(kept, np.arange(3))


def test_robust_pipelines_are_registered():
    """Состав стенда в оркестраторе должен существовать в реестре."""
    from deepfakes.pipelines.registry import REGISTRY
    from scripts.run_study import ROBUST_PIPELINES, TUNE_PIPELINES

    for name in ROBUST_PIPELINES + TUNE_PIPELINES:
        assert name in REGISTRY, f"{name} нет в реестре пайплайнов"


def test_tune_pipelines_have_search_spaces():
    """Стадия tune без пространства поиска упала бы уже на кластере."""
    from deepfakes.tuning.optuna_runner import PIPELINE_SPACES
    from scripts.run_study import TUNE_PIPELINES

    missing = [p for p in TUNE_PIPELINES if p not in PIPELINE_SPACES]
    assert not missing, f"нет пространства поиска: {missing}"


def test_subset_narrows_stage_and_rejects_typos():
    """Разбиение стадии на несколько sbatch не должно молча съедать опечатку."""
    from scripts.run_study import ROBUST_PIPELINES, _subset

    assert _subset(ROBUST_PIPELINES, None) == ROBUST_PIPELINES
    assert _subset(ROBUST_PIPELINES, "freq4, tex2") == ["freq4", "tex2"]
    with pytest.raises(SystemExit):
        _subset(ROBUST_PIPELINES, "freq4,opechatka")
