"""Smoke tests для новых пайплайнов NOISE-1 и ML-6."""

from __future__ import annotations

import pytest

from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.ml6_hybrid import ML6HybridPipeline
from deepfakes.pipelines.noise1 import Noise1Pipeline


@pytest.fixture(params=["logreg", "random_forest", "xgboost"])
def model_spec(request):
    return ModelSpec(name=request.param, params={})


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


def test_noise1_nlmeans(model_spec, tiny_splits):
    cfg = {
        "img_size": 64,
        "denoise_method": "nlmeans",
        "denoise_strength": 7.0,
        "use_srm": True,
        "use_residual": True,
        "patch_grid": 2,
    }
    _check(Noise1Pipeline, cfg, model_spec, tiny_splits)


def test_noise1_median_no_srm(model_spec, tiny_splits):
    cfg = {
        "img_size": 64,
        "denoise_method": "median",
        "denoise_strength": 3.0,
        "use_srm": False,
        "use_residual": True,
        "patch_grid": 1,
    }
    _check(Noise1Pipeline, cfg, model_spec, tiny_splits)


def test_ml6_hybrid(model_spec, tiny_splits):
    cfg = {
        "img_size": 64,
        "color_patch_sizes": [16, 32],
        "colorspace": "YCbCr",
        "color_full_stats": True,
        "lbp_radii": [1, 2],
        "lbp_method": "uniform",
    }
    _check(ML6HybridPipeline, cfg, model_spec, tiny_splits)
