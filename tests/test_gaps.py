"""Train-val gap monitoring."""

from __future__ import annotations

import numpy as np

from deepfakes.eval.metrics import compute_report
from deepfakes.pipelines.base import BasePipeline


def _rep(auc_pos: bool = True):
    rng = np.random.default_rng(0)
    n = 100
    y = rng.integers(0, 2, size=n)
    s = y.astype(float) * (1.0 if auc_pos else -1.0) + rng.normal(0, 0.3, n)
    s = (s - s.min()) / (s.max() - s.min())
    return compute_report(y, s)


def test_gap_train_overfit_positive():
    """Train > Val → gap_train_val_* > 0."""
    train = _rep(auc_pos=True)
    val = compute_report(np.array([0, 0, 1, 1]), np.array([0.4, 0.6, 0.4, 0.6]))
    gaps = BasePipeline.compute_gaps(train, val)
    assert "gap_train_val_auc" in gaps
    assert gaps["gap_train_val_auc"] > 0


def test_gap_all_three_keys():
    r = _rep()
    gaps = BasePipeline.compute_gaps(r, r, r)
    assert {
        "gap_train_val_auc",
        "gap_train_val_acc",
        "gap_train_val_f1",
        "gap_val_test_auc",
        "gap_val_test_acc",
        "gap_train_test_auc",
    }.issubset(gaps)


def test_gap_handles_none():
    assert BasePipeline.compute_gaps(None, None) == {}
    assert "gap_train_val_auc" in BasePipeline.compute_gaps(_rep(), _rep())
