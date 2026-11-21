"""DeepFeatureExtractor: импорт безопасен без torch; извлечение — при наличии."""

from __future__ import annotations

import glob

import numpy as np
import pytest


def test_import_without_torch_is_safe():
    # сам модуль и фабрика ROB-1 обязаны импортироваться без torch/timm
    from deepfakes.features.deep import DeepFeatureExtractor
    from deepfakes.pipelines.rob1_fusion import _FACTORY

    assert "deep" in _FACTORY
    ex = DeepFeatureExtractor(backbone="vit_tiny_patch16_224")
    assert ex.n_jobs == 1  # процессный параллелизм для модели отключён


def test_missing_torch_raises_helpful_error(tiny_dataset):
    pytest.importorskip("torch", reason="этот тест — только БЕЗ torch")
    pytest.skip("torch установлен — ветка ошибки не проверяется")


def test_deep_extraction_if_torch_available(tiny_dataset):
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("timm")
    from deepfakes.features.deep import DeepFeatureExtractor

    paths = np.array(sorted(glob.glob(tiny_dataset["real_dir"] + "/*.png"))[:5], dtype=object)
    ex = DeepFeatureExtractor(
        backbone="vit_tiny_patch16_224", img_size=224, pretrained=False, verbose=False
    )
    F = ex.transform(paths)
    assert F.shape[0] == 5
    assert F.shape[1] > 100  # эмбеддинг ViT-tiny = 192
    names = ex.get_feature_names()
    assert names is not None and len(names) == F.shape[1]
    assert names[0].startswith("deep_vit_tiny")
