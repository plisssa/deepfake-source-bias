"""Smoke-тесты для всех новых пайплайнов (по 2 из каждой группы)."""

from __future__ import annotations

import pytest

from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.col1 import Col1Pipeline
from deepfakes.pipelines.col3 import Col3Pipeline
from deepfakes.pipelines.freq2 import Freq2Pipeline
from deepfakes.pipelines.freq3 import Freq3Pipeline
from deepfakes.pipelines.grad1 import Grad1Pipeline
from deepfakes.pipelines.grad3 import Grad3Pipeline
from deepfakes.pipelines.ml7_freq_lbp import ML7FreqLBPPipeline
from deepfakes.pipelines.ml8_anomaly import ML8AnomalyPipeline
from deepfakes.pipelines.noise2 import Noise2Pipeline
from deepfakes.pipelines.noise3 import Noise3Pipeline
from deepfakes.pipelines.rob1_fusion import Rob1FusionPipeline
from deepfakes.pipelines.tex4 import Tex4Pipeline
from deepfakes.pipelines.tex6 import Tex6Pipeline


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


def test_grad1_sobel(model_spec, tiny_splits):
    _check(
        Grad1Pipeline,
        {"img_size": 64, "ksize": 3, "patch_grid": 2, "n_orient_bins": 9},
        model_spec,
        tiny_splits,
    )


def test_grad3_canny(model_spec, tiny_splits):
    _check(
        Grad3Pipeline,
        {
            "img_size": 64,
            "low_threshold": 30,
            "high_threshold": 90,
            "patch_grid": 2,
            "n_orient_bins": 9,
        },
        model_spec,
        tiny_splits,
    )


def test_col1_histograms(model_spec, tiny_splits):
    _check(
        Col1Pipeline,
        {"img_size": 64, "n_bins": 16, "colorspace": "HSV", "patch_grid": 1},
        model_spec,
        tiny_splits,
    )


def test_col3_ycbcr_chi(model_spec, tiny_splits):
    _check(
        Col3Pipeline,
        {"img_size": 64, "n_bins": 16, "patch_grid": 2, "full_stats": True},
        model_spec,
        tiny_splits,
    )


def test_freq2_dct(model_spec, tiny_splits):
    _check(
        Freq2Pipeline, {"img_size": 64, "block_size": 8, "log_scale": True}, model_spec, tiny_splits
    )


def test_freq3_wavelet(model_spec, tiny_splits):
    _check(Freq3Pipeline, {"img_size": 64, "wavelet": "db2", "level": 2}, model_spec, tiny_splits)


def test_tex4_hog(model_spec, tiny_splits):
    _check(
        Tex4Pipeline,
        {"img_size": 64, "orientations": 9, "pixels_per_cell": 8, "cells_per_block": 2},
        model_spec,
        tiny_splits,
    )


def test_tex6_glcm(model_spec, tiny_splits):
    _check(
        Tex6Pipeline,
        {"img_size": 64, "distances": [1, 2], "angles": None, "levels": 16, "patch_grid": 1},
        model_spec,
        tiny_splits,
    )


def test_noise2_blockiness(model_spec, tiny_splits):
    _check(
        Noise2Pipeline, {"img_size": 64, "block_size": 8, "patch_grid": 2}, model_spec, tiny_splits
    )


def test_noise3_prnu(model_spec, tiny_splits):
    _check(
        Noise3Pipeline,
        {"img_size": 64, "denoise_method": "median", "denoise_strength": 3.0, "patch_grid": 2},
        model_spec,
        tiny_splits,
    )


def test_ml7_freq_lbp(model_spec, tiny_splits):
    cfg = {"img_size": 64, "freq_n_bins": 6, "lbp_radii": [1, 2], "lbp_method": "uniform"}
    _check(ML7FreqLBPPipeline, cfg, model_spec, tiny_splits)


def test_ml8_anomaly_oneclass(tiny_splits):
    """One-class — модель строит сам, model_spec для совместимости только."""
    cfg = {
        "features": "prnu",
        "img_size": 64,
        "denoise_method": "median",
        "denoise_strength": 3.0,
        "patch_grid": 2,
        "oc_params": {"nu": 0.2, "kernel": "rbf", "gamma": "scale"},
    }
    pipe = ML8AnomalyPipeline.from_cfg(cfg)
    pipe.fit(
        tiny_splits["train"]["X"],
        tiny_splits["train"]["y"],
        X_val_paths=tiny_splits["val"]["X"],
        y_val=tiny_splits["val"]["y"],
    )
    probs = pipe.predict_proba(tiny_splits["test"]["X"])
    assert probs.shape == (len(tiny_splits["test"]["X"]),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_rob1_fusion(model_spec, tiny_splits):
    cfg = {
        "components": [
            {"kind": "freq", "cfg": {"img_size": 64, "n_bins": 6}},
            {"kind": "lbp", "cfg": {"img_size": 64, "radii": [1, 2], "method": "uniform"}},
        ],
        "fusion": {"scale": True, "variance_threshold": 0.0, "pca_components": 10},
    }
    _check(Rob1FusionPipeline, cfg, model_spec, tiny_splits)
