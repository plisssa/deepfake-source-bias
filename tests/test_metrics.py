"""Метрики, threshold tuning, bootstrap."""

from __future__ import annotations

import numpy as np

from deepfakes.eval.metrics import (
    bootstrap_ci,
    compute_eer,
    compute_report,
    tune_threshold,
)


def test_compute_report_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.95])
    rep = compute_report(y, s, threshold=0.5)
    assert rep.accuracy == 1.0
    assert rep.roc_auc == 1.0
    assert rep.f1 == 1.0
    assert rep.confusion_matrix == [[2, 0], [0, 2]]


def test_compute_report_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    rep = compute_report(y, s)
    assert 0.3 <= rep.roc_auc <= 0.7
    assert rep.n == 200
    assert rep.n_pos + rep.n_neg == 200


def test_eer_symmetric():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate(
        [np.random.RandomState(0).beta(2, 5, 50), np.random.RandomState(1).beta(5, 2, 50)]
    )
    eer, thr = compute_eer(y, s)
    assert 0.0 <= eer <= 1.0
    assert 0.0 <= thr <= 1.0


def test_threshold_tuning():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.45, 0.55, 0.7, 0.9])
    t_f1 = tune_threshold(y, s, objective="f1")
    t_youden = tune_threshold(y, s, objective="youden")
    t_eer = tune_threshold(y, s, objective="eer")
    for t in (t_f1, t_youden, t_eer):
        assert 0.0 <= t <= 1.0


def test_bootstrap_ci():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    s = y + rng.normal(0, 0.5, size=500)
    point, lo, hi = bootstrap_ci(y, s, roc_auc_score, n_resamples=200, seed=42)
    assert lo <= point <= hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
