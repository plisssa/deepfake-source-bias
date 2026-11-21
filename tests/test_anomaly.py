"""OneClassPredictor (ML-8 model): обучение на real-only и predict_proba."""

from __future__ import annotations

import numpy as np
import pytest

from deepfakes.models.anomaly import OneClassPredictor


def test_one_class_only_real_in_train():
    rng = np.random.default_rng(0)
    X_real = rng.normal(0, 1, size=(80, 5))
    X_fake = rng.normal(3, 1, size=(80, 5))  # «выбросы»
    X = np.vstack([X_real, X_fake])
    y = np.array([0] * 80 + [1] * 80)
    clf = OneClassPredictor(nu=0.1, kernel="rbf")
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (160, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # Среднее P(fake) на синтетических fake должно быть выше, чем на real
    mean_real = proba[y == 0, 1].mean()
    mean_fake = proba[y == 1, 1].mean()
    assert mean_fake > mean_real, f"fake≈{mean_fake:.3f} not greater than real≈{mean_real:.3f}"


def test_one_class_requires_real():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(20, 5))
    y = np.ones(20)  # все fake
    with pytest.raises(ValueError):
        OneClassPredictor().fit(X, y)
