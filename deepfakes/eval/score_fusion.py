"""ROB-2: Score-level fusion нескольких пайплайнов (weighted ensemble).

Поддерживаются три режима:
  1. uniform  — простое усреднение probabilities.
  2. logreg   — логистическая регрессия как meta-learner (stacking).
  3. optuna   — подбор весов через Optuna на val (sum(w)=1, w>=0)
                с целью максимизации val ROC-AUC.

Все режимы возвращают finalized predictor (callable: X → probabilities).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

FusionMode = Literal["uniform", "logreg", "optuna"]


@dataclass
class FusionResult:
    mode: FusionMode
    weights: np.ndarray | None  # для uniform/optuna
    meta_model: LogisticRegression | None  # для logreg
    val_auc: float
    component_names: list[str]


def _check_inputs(score_matrix: np.ndarray, y: np.ndarray) -> None:
    if score_matrix.ndim != 2:
        raise ValueError(f"score_matrix must be 2D (n_samples, n_models), got {score_matrix.shape}")
    if len(y) != score_matrix.shape[0]:
        raise ValueError(
            f"len(y)={len(y)} mismatches score_matrix.shape[0]={score_matrix.shape[0]}"
        )


def fuse_uniform(val_scores: np.ndarray, val_y: np.ndarray, names: list[str]) -> FusionResult:
    _check_inputs(val_scores, val_y)
    w = np.ones(val_scores.shape[1]) / val_scores.shape[1]
    pred = val_scores @ w
    return FusionResult("uniform", w, None, float(roc_auc_score(val_y, pred)), names)


def fuse_logreg(
    val_scores: np.ndarray, val_y: np.ndarray, names: list[str], seed: int = 42
) -> FusionResult:
    """val_auc — out-of-fold (5-fold CV) оценка, а не in-sample: fit и оценка
    на одном и том же val систематически завышали заявленный AUC фьюжна."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    _check_inputs(val_scores, val_y)
    meta = LogisticRegression(C=1.0, max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(meta, val_scores, val_y, cv=cv, method="predict_proba")[:, 1]
    val_auc = float(roc_auc_score(val_y, oof))
    meta.fit(val_scores, val_y)  # финальная мета-модель — на всём val
    return FusionResult("logreg", None, meta, val_auc, names)


def fuse_optuna(
    val_scores: np.ndarray,
    val_y: np.ndarray,
    names: list[str],
    n_trials: int = 100,
    seed: int = 42,
) -> FusionResult:
    """Найти w_i >= 0, sum=1, максимизирующие AUC.

    Веса оптимизируются на одной стратифицированной половине val, а val_auc
    считается по итоговым весам на удержанной половине — иначе 100 trials
    переобучались под val, и заявленный AUC был in-sample."""
    import optuna
    from sklearn.model_selection import train_test_split

    _check_inputs(val_scores, val_y)
    n_models = val_scores.shape[1]
    idx_fit, idx_hold = train_test_split(
        np.arange(len(val_y)), test_size=0.5, stratify=val_y, random_state=seed
    )
    fit_scores, fit_y = val_scores[idx_fit], val_y[idx_fit]
    hold_scores, hold_y = val_scores[idx_hold], val_y[idx_hold]

    def objective(trial: optuna.Trial) -> float:
        raw = np.array(
            [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)], dtype=np.float64
        )
        s = raw.sum()
        if s < 1e-9:
            return 0.0
        w = raw / s
        pred = fit_scores @ w
        return float(roc_auc_score(fit_y, pred))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    raw = np.array([study.best_params[f"w_{i}"] for i in range(n_models)])
    w = raw / max(raw.sum(), 1e-9)
    holdout_auc = float(roc_auc_score(hold_y, hold_scores @ w))
    return FusionResult("optuna", w, None, holdout_auc, names)


def apply_fusion(result: FusionResult, test_scores: np.ndarray) -> np.ndarray:
    """Применить найденный fusion к новой матрице скорингов (n_samples, n_models)."""
    if result.meta_model is not None:
        return result.meta_model.predict_proba(test_scores)[:, 1]
    assert result.weights is not None
    return test_scores @ result.weights


def collect_scores(
    pipelines_and_specs: list[tuple[str, dict, dict]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    cache_dir: str | None = None,
    names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Обучить каждый из пайплайнов на train и собрать val/test предсказания.

    Args:
        pipelines_and_specs: список (pipeline_name, pipeline_cfg_dict, model_spec_dict),
            где model_spec_dict = {'name': ..., 'params': {...}}.

    Returns:
        (val_score_matrix, test_score_matrix, names) — формы (n_samples, n_models).
    """
    from deepfakes.models.classical import ModelSpec
    from deepfakes.pipelines.registry import get_pipeline_class

    if names is None:
        names = [f"{p[0]}__{p[2]['name']}" for p in pipelines_and_specs]

    val_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    for (pname, pcfg, mspec), label in zip(pipelines_and_specs, names):
        log.info(f"[fusion] training {label}")
        cls = get_pipeline_class(pname)
        spec = ModelSpec(name=mspec["name"], params=dict(mspec.get("params", {})))
        pipe = cls.from_cfg(pcfg, spec, cache_dir=cache_dir)
        pipe.fit(X_train, y_train)
        val_cols.append(pipe.predict_proba(X_val))
        test_cols.append(pipe.predict_proba(X_test))
    return np.column_stack(val_cols), np.column_stack(test_cols), names
