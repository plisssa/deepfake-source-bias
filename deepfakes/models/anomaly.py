"""ML-8 helper: One-Class SVM как «классификатор» бинарной задачи.

Обучается только на real-классе (y==0). predict_proba возвращает синтетическую
вероятность fake = σ((−decision_function − μ) / σ̂), где μ,σ̂ оцениваются по
calibration-set (fit-time).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class OneClassPredictor(BaseEstimator, ClassifierMixin):
    """One-Class SVM с predict_proba и стандартной X-предобработкой."""

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        with_scaler: bool = True,
    ):
        self.nu = float(nu)
        self.kernel = kernel
        self.gamma = gamma
        self.with_scaler = with_scaler
        self.scaler_: StandardScaler | None = None
        self.model_: OneClassSVM | None = None
        self.mu_: float = 0.0
        self.sigma_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        X_real = X[y == 0]
        if len(X_real) == 0:
            raise ValueError("ML-8 / OneClassPredictor: no real samples (y==0) in training set")
        if self.with_scaler:
            self.scaler_ = StandardScaler()
            X_real = self.scaler_.fit_transform(X_real)
        self.model_ = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model_.fit(X_real)
        # Калибровка μ/σ̂ на training scores
        scores = -self.model_.decision_function(X_real)
        self.mu_ = float(scores.mean())
        self.sigma_ = float(scores.std() + 1e-6)
        self.classes_ = np.array([0, 1])
        return self

    def _scaled(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is not None:
            return self.scaler_.transform(X)
        return X

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.decision_function(self._scaled(np.asarray(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores_fake = -self.decision_function(X)
        p_fake = expit((scores_fake - self.mu_) / self.sigma_)
        return np.column_stack([1.0 - p_fake, p_fake])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
