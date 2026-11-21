"""ROB-2: тесты для score_fusion (uniform / logreg / optuna)."""

from __future__ import annotations

import numpy as np

from deepfakes.eval.score_fusion import apply_fusion, fuse_logreg, fuse_optuna, fuse_uniform


def _make_synthetic(n: int = 200, n_models: int = 3, noise: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    base = y.astype(float)
    val_scores = np.stack(
        [np.clip(base + rng.normal(0, noise, n), 0, 1) for _ in range(n_models)], axis=1
    )
    test_scores = np.stack(
        [np.clip(base + rng.normal(0, noise, n), 0, 1) for _ in range(n_models)], axis=1
    )
    return val_scores, test_scores, y


def test_uniform_fusion_better_than_chance():
    val, test, y = _make_synthetic()
    res = fuse_uniform(val, y, ["a", "b", "c"])
    assert res.mode == "uniform"
    assert np.isclose(res.weights.sum(), 1.0)
    assert res.val_auc > 0.7
    pred = apply_fusion(res, test)
    assert pred.shape == (len(y),)


def test_logreg_fusion_runs():
    val, test, y = _make_synthetic()
    res = fuse_logreg(val, y, ["a", "b", "c"])
    assert res.mode == "logreg"
    assert res.meta_model is not None
    pred = apply_fusion(res, test)
    assert pred.shape == (len(y),)


def test_optuna_fusion_constraints():
    val, test, y = _make_synthetic()
    res = fuse_optuna(val, y, ["a", "b", "c"], n_trials=15, seed=0)
    assert res.mode == "optuna"
    assert res.weights is not None
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
    assert (res.weights >= 0).all()
    pred = apply_fusion(res, test)
    assert pred.shape == (len(y),)
