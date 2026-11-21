"""Обвязка для cross-validation: usable как для классики, так и для DL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from deepfakes.eval.metrics import ClassificationReport, compute_report
from deepfakes.io.splits import kfold_iter
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FoldResult:
    fold: int
    report: ClassificationReport
    train_size: int
    val_size: int


def run_kfold(
    X: np.ndarray,
    y: np.ndarray,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 5,
    seed: int = 42,
    threshold: float = 0.5,
    groups: np.ndarray | None = None,
) -> tuple[list[FoldResult], dict[str, float]]:
    """Прогнать stratified (group-aware при наличии groups) K-fold.

    fit_predict_fn(X_train, y_train, X_val, y_val) возвращает либо y_val_score,
    либо (y_val_score, tuned_threshold) — тогда отчёт фолда считается порогом,
    тюненным на fold-val (сопоставимо с holdout-протоколом).
    """
    fold_reports: list[FoldResult] = []
    for fold_idx, train_idx, val_idx in kfold_iter(
        X, y, n_splits=n_splits, seed=seed, groups=groups
    ):
        log.info(f"  ▶ fold {fold_idx + 1}/{n_splits} (train={len(train_idx)}, val={len(val_idx)})")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        result = fit_predict_fn(X_train, y_train, X_val, y_val)
        if isinstance(result, tuple):
            y_score, fold_threshold = result
        else:
            y_score, fold_threshold = result, threshold
        report = compute_report(y_val, y_score, threshold=fold_threshold)
        fold_reports.append(
            FoldResult(
                fold=fold_idx,
                report=report,
                train_size=len(train_idx),
                val_size=len(val_idx),
            )
        )

    return fold_reports, aggregate_folds(fold_reports)


# Не метрики: усреднять их по фолдам бессмысленно (порог) или вводит в
# заблуждение (объёмы выборок).
_NON_METRIC_COLS = {"confusion_matrix", "threshold", "n", "n_pos", "n_neg"}


def aggregate_folds(folds: list[FoldResult]) -> dict[str, float]:
    """Mean ± std по фолдам для всех скалярных метрик."""
    rows = [f.report.to_dict() for f in folds]
    df = pd.DataFrame(rows)
    numeric_cols = [
        c for c in df.columns if c not in _NON_METRIC_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    agg: dict[str, float] = {}
    for c in numeric_cols:
        agg[f"{c}_mean"] = float(df[c].mean())
        agg[f"{c}_std"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return agg


def folds_to_csv(folds: list[FoldResult], out_path: str) -> None:
    rows = []
    for f in folds:
        row = {"fold": f.fold, "train_size": f.train_size, "val_size": f.val_size}
        row.update(f.report.to_dict())
        # convert lists to strings for CSV friendliness
        row["confusion_matrix"] = str(row["confusion_matrix"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
