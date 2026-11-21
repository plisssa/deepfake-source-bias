"""Сохранённый пайплайн загружается обратно и предсказывает ТО ЖЕ самое.

Регрессия 29.09.2026: `save()` писал словарь, а `load()` не существовало —
пересчёт предсказаний без переобучения падал с
`'dict' object has no attribute 'predict_proba'`."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline
from deepfakes.pipelines.registry import get_pipeline_class


@pytest.fixture()
def tiny(tmp_path):
    rng = np.random.default_rng(0)
    paths, y = [], []
    for i in range(40):
        arr = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
        if i % 2:
            arr[::2, ::2] = 255  # простой «артефакт» fake-класса
        p = tmp_path / f"{i:03d}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
        y.append(i % 2)
    return np.array(paths, dtype=object), np.array(y)


def test_roundtrip_gives_identical_predictions(tiny, tmp_path):
    from omegaconf import OmegaConf

    X, y = tiny
    cfg = OmegaConf.create(
        {
            "name": "freq1",
            "img_size": 48,
            "radial_bins": 8,
            "angular_bins": 4,
            "window": "hann",
            "include_phase": False,
        }
    )
    cls = get_pipeline_class("freq1")
    pipe = cls.from_cfg(cfg, ModelSpec(name="logreg", params={}), cache_dir=str(tmp_path / "c"))
    pipe.fit(X, y)
    before = pipe.predict_proba(X)

    path = pipe.save(tmp_path / "pipeline.joblib")
    loaded = BasePipeline.load(path, cache_dir=str(tmp_path / "c"))
    after = loaded.predict_proba(X)

    np.testing.assert_array_equal(before, after)
    assert loaded.threshold == pipe.threshold
    assert loaded.pipeline_id == pipe.pipeline_id


def test_load_rejects_foreign_file(tmp_path):
    import joblib

    p = tmp_path / "x.joblib"
    joblib.dump({"something": 1}, p)
    with pytest.raises(ValueError):
        BasePipeline.load(p)
