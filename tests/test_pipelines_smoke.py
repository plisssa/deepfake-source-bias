"""Smoke-tests: каждый пайплайн обучается и предсказывает на 80 изображениях."""

from __future__ import annotations

import pytest

from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.col2 import Col2Pipeline
from deepfakes.pipelines.freq4 import Freq4Pipeline
from deepfakes.pipelines.grad4 import Grad4Pipeline
from deepfakes.pipelines.tex2 import Tex2Pipeline


@pytest.fixture(params=["logreg", "random_forest", "xgboost"])
def model_spec(request):
    return ModelSpec(name=request.param, params={})


def _make_cfg_dict(extra: dict) -> dict:
    return extra


def _check(pipe_cls, pipeline_cfg, model_spec, tiny_splits):
    pipe = pipe_cls.from_cfg(pipeline_cfg, model_spec, cache_dir=None)
    artifacts = pipe.fit(
        tiny_splits["train"]["X"],
        tiny_splits["train"]["y"],
        X_val_paths=tiny_splits["val"]["X"],
        y_val=tiny_splits["val"]["y"],
    )
    assert artifacts.feature_dim > 0
    probs = pipe.predict_proba(tiny_splits["test"]["X"])
    assert probs.shape == (len(tiny_splits["test"]["X"]),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_col2(model_spec, tiny_splits):
    cfg = {"img_size": 64, "patch_sizes": (16, 32), "colorspace": "RGB", "full_stats": True}
    _check(Col2Pipeline, cfg, model_spec, tiny_splits)


def test_freq4(model_spec, tiny_splits):
    cfg = {
        "img_size": 64,
        "n_bins": 6,
        "include_slope": True,
        "include_flatness": True,
        "include_basic_stats": True,
        "use_gan_augment": False,
    }
    _check(Freq4Pipeline, cfg, model_spec, tiny_splits)


def test_grad4_patch(model_spec, tiny_splits):
    cfg = {"mode": "patch", "img_size": 64, "sigma": 0.7, "patch_size": 16, "full_stats": True}
    _check(Grad4Pipeline, cfg, model_spec, tiny_splits)


def test_tex2_uniform(model_spec, tiny_splits):
    cfg = {"img_size": 64, "radii": [1, 2], "method": "uniform"}
    _check(Tex2Pipeline, cfg, model_spec, tiny_splits)
