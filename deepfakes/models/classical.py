"""Classical classifiers and the hyperparameters used by each pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator

ModelName = Literal["logreg", "random_forest", "xgboost", "lightgbm", "catboost", "svm", "mlp"]


@dataclass
class ModelSpec:
    """Спецификация модели для бенчмарка."""

    name: ModelName
    params: dict[str, Any]


def build_model(spec: ModelSpec, n_features: int | None = None) -> BaseEstimator:
    """Фабрика моделей с разумными дефолтами и поддержкой scale_pos_weight для XGBoost."""
    if spec.name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        default = dict(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
        default.update(spec.params)
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**default))])

    if spec.name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        default = dict(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            oob_score=True,
        )
        default.update(spec.params)
        return RandomForestClassifier(**default)

    if spec.name == "xgboost":
        from xgboost import XGBClassifier

        default = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        default.update(spec.params)
        return XGBClassifier(**default)

    if spec.name == "lightgbm":
        from lightgbm import LGBMClassifier

        default = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        default.update(spec.params)
        return LGBMClassifier(**default)

    if spec.name == "catboost":
        from catboost import CatBoostClassifier

        default = dict(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
            auto_class_weights="Balanced",
        )
        default.update(spec.params)
        return CatBoostClassifier(**default)

    if spec.name == "svm":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        default = dict(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
        default.update(spec.params)
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(**default))])

    if spec.name == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        default = dict(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            max_iter=300,
            random_state=42,
        )
        default.update(spec.params)
        return Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(**default))])

    raise ValueError(f"Unknown model name: {spec.name}")


def predict_proba_safe(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Универсальный predict_proba: возвращает вероятность класса 1."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        return expit(model.decision_function(X))
    return model.predict(X).astype(float)
