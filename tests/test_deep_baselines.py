"""Обученные бейзлайны: вход без интерполяции, конфиги, отличимость вариантов.

Сами модели здесь не грузятся (torch локально нет) — проверяется всё, что
определяет честность сравнения ДО модели: кроп, параметры, теги."""

from __future__ import annotations

import os

import numpy as np
import pytest

from deepfakes.features.deep import DeepFeatureExtractor, center_crop_or_pad


def test_center_crop_takes_pixels_as_is():
    img = np.arange(256 * 256 * 3, dtype=np.int64).reshape(256, 256, 3) % 251
    out = center_crop_or_pad(img, 224)
    assert out.shape == (224, 224, 3)
    np.testing.assert_array_equal(out, img[16:240, 16:240])  # ни одного пересчитанного пикселя


def test_small_image_is_padded_not_resized():
    img = np.random.default_rng(0).integers(0, 256, (200, 210, 3), dtype=np.uint8)
    out = center_crop_or_pad(img, 224)
    assert out.shape == (224, 224, 3)
    # исходные пиксели присутствуют без изменений
    assert any(
        np.array_equal(out[r : r + 200, c : c + 210], img) for r in range(25) for c in range(15)
    )


def test_bad_modes_rejected():
    with pytest.raises(ValueError):
        DeepFeatureExtractor(resize_mode="stretch")
    with pytest.raises(ValueError):
        DeepFeatureExtractor(normalize="none")


def test_new_params_enter_cache_key():
    """Смена кропа/нормализации обязана менять ключ кэша признаков."""
    a = DeepFeatureExtractor(backbone="x", resize_mode="resize", normalize="imagenet")
    b = DeepFeatureExtractor(backbone="x", resize_mode="center_crop", normalize="model")
    pa, pb = a.get_params(deep=False), b.get_params(deep=False)
    assert pa["resize_mode"] != pb["resize_mode"]
    assert pa["normalize"] != pb["normalize"]


@pytest.mark.parametrize(
    "cfg_name,backbone",
    [
        ("clip_l14_lp", "vit_large_patch14_clip_224.openai"),
        ("dinov2_b14_lp", "vit_base_patch14_dinov2.lvd142m"),
    ],
)
def test_configs_build_pipeline_and_stay_distinguishable(cfg_name, backbone):
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    from deepfakes.models.classical import ModelSpec
    from deepfakes.pipelines.registry import get_pipeline_class
    from deepfakes.tracking.mlflow_setup import pipeline_variant

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"pipeline={cfg_name}", "model=logreg"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        pipe = get_pipeline_class(cfg.pipeline.name).from_cfg(
            cfg.pipeline, ModelSpec(name="logreg", params={})
        )
        assert pipe.extractor.backbone == backbone
        assert pipe.extractor.resize_mode == "center_crop"
        assert pipe.extractor.normalize == "model"
        assert cfg.pipeline.name == "deep_lp"  # класс общий…
        assert pipeline_variant(cfg) == cfg_name  # …а в MLflow варианты различимы


def test_iter_images_ordered_keeps_order_and_marks_failures():
    """Параллельное чтение обязано отдавать кадры в ИСХОДНОМ порядке."""
    import random
    import time

    from deepfakes.features.deep import iter_images_ordered

    paths = [f"img_{i}" for i in range(200)]

    def read(p):
        time.sleep(random.random() * 0.002)  # потоки завершаются вразнобой
        if p == "img_7":
            raise OSError("битый файл")
        return int(p.split("_")[1])

    out = list(iter_images_ordered(paths, read, n_threads=8, prefetch=16))
    assert [i for i, _ in out] == list(range(200)), "порядок нарушен"
    assert out[7][1] is None, "нечитаемый файл — None, а не исключение"
    assert all(v == i for i, v in out if i != 7), "кадр не совпадает со своим путём"


def test_iter_images_ordered_bounds_prefetch():
    """В полёте не больше prefetch чтений — память не растёт на всю выборку."""
    import threading

    from deepfakes.features.deep import iter_images_ordered

    live, peak = 0, 0
    lock = threading.Lock()
    gate = threading.Event()

    def read(p):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        gate.wait(0.001)
        with lock:
            live -= 1
        return p

    for _ in iter_images_ordered([str(i) for i in range(300)], read, n_threads=4, prefetch=10):
        pass
    assert peak <= 10


def test_amp_enters_cache_key():
    a = DeepFeatureExtractor(backbone="x", amp=True).get_params(deep=False)
    b = DeepFeatureExtractor(backbone="x", amp=False).get_params(deep=False)
    assert a["amp"] != b["amp"]


def test_extractor_pickles_without_torch_and_weights():
    """Регрессия: `cannot pickle 'module' object` при сохранении пайплайна."""
    import math
    import pickle

    ext = DeepFeatureExtractor(backbone="x")
    ext._torch = math  # любой модуль — именно он и не сериализовался
    ext._model = object()  # «веса»
    restored = pickle.loads(pickle.dumps(ext))
    assert restored._torch is None and restored._model is None, "бэкбон не должен сохраняться"
    assert restored.backbone == "x" and restored.resize_mode == ext.resize_mode
    assert ext._torch is math, "сериализация не должна портить исходный объект"


def test_old_pickle_restores_historical_behaviour():
    """Объект, сохранённый старым кодом, обязан вести себя ПО-СТАРОМУ.

    Иначе исторический ViT-гибрид (обучен на ImageNet-нормализации, ресайзе и
    fp32) получил бы при загрузке новый вход и молча выдал другие числа."""
    import pickle

    ext = DeepFeatureExtractor(backbone="x", resize_mode="center_crop", normalize="model")
    state = ext.__getstate__()
    for k in ("resize_mode", "normalize", "model_img_size", "amp", "_mean", "_std"):
        state.pop(k)  # так выглядел __dict__ до 29.09.2026
    old = DeepFeatureExtractor.__new__(DeepFeatureExtractor)
    old.__setstate__(state)
    restored = pickle.loads(pickle.dumps(old))
    assert restored.resize_mode == "resize"
    assert restored.normalize == "imagenet"
    assert restored.amp is False
    np.testing.assert_allclose(restored._mean, [0.485, 0.456, 0.406])


def test_new_pickle_keeps_its_own_settings():
    import pickle

    ext = DeepFeatureExtractor(backbone="x", resize_mode="center_crop", normalize="model", amp=True)
    r = pickle.loads(pickle.dumps(ext))
    assert (r.resize_mode, r.normalize, r.amp) == ("center_crop", "model", True)
