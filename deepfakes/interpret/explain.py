"""Объяснимость: SHAP (для деревьев) + permutation importance (модель-агностик).

Оба отчёта следуют контракту RankingReport из feature_selection: полный CSV
важностей + PNG-plot. Дополнительно — combined_ranking_report (PFI+SHAP
Ensemble): усреднение нормированных рангов двух методов.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepfakes.interpret.feature_selection import RankingReport, default_feature_names
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


def _is_tree_model(model: Any) -> bool:
    cls = model.__class__.__name__.lower()
    return any(
        k in cls for k in ("xgb", "lgbm", "lightgbm", "catboost", "randomforest", "gradientboost")
    )


def _binary_class_shap(sv: Any) -> np.ndarray:
    """Привести shap_values к 2D-матрице (n, features) для класса 1.

    Новый API SHAP для деревьев возвращает 3D ndarray (n, features, n_classes);
    старый — список из n_classes массивов. Оба случая срезаем к классу 1.
    """
    if isinstance(sv, list):
        return sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def shap_summary(
    model: Any,
    X: np.ndarray,
    out_path: str | Path,
    n_samples: int = 200,
    feature_names: list[str] | None = None,
) -> RankingReport | None:
    """SHAP summary plot + CSV mean(|shap|) по признакам. None — если не вышло.

    Ошибки SHAP не роняют запуск, но логируются warning-ом с текстом ошибки.
    """
    try:
        import shap
    except ImportError:
        log.warning("SHAP not installed — skipping")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Xs = X[:n_samples]
    underlying = model
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        underlying = model.named_steps["clf"]
        Xs = model[:-1].transform(Xs)

    try:
        if _is_tree_model(underlying):
            explainer = shap.TreeExplainer(underlying)
            sv_plot = _binary_class_shap(explainer.shap_values(Xs))
        else:
            explainer = shap.Explainer(underlying.predict_proba, Xs)
            sv_plot = explainer(Xs).values
            if sv_plot.ndim == 3:
                sv_plot = sv_plot[..., 1]

        names = feature_names or default_feature_names(Xs.shape[1])
        shap.summary_plot(sv_plot, Xs, show=False, feature_names=names)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        df = pd.DataFrame(
            {"feature": names, "mean_abs_shap": np.abs(sv_plot).mean(axis=0)}
        ).sort_values("mean_abs_shap", ascending=False)
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return RankingReport(csv_path, out_path, top_k=len(names))
    except Exception as e:
        log.warning(f"SHAP failed ({type(e).__name__}): {e}")
        return None


def permutation_importance_plot(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    out_path: str | Path,
    n_repeats: int = 5,
    feature_names: list[str] | None = None,
    top_k: int = 30,
) -> RankingReport | None:
    """Permutation importance (модель-агностик): полный CSV + bar-plot топ-K."""
    from sklearn.inspection import permutation_importance

    try:
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=42, scoring="roc_auc", n_jobs=-1
        )
    except Exception as e:
        log.warning(f"permutation_importance failed: {e}")
        return None

    names = feature_names or default_feature_names(X.shape[1])
    df = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    top = df.head(top_k)
    fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.18)))
    ax.barh(top["feature"][::-1], top["importance_mean"][::-1])
    ax.set_xlabel("Mean drop in ROC-AUC")
    ax.set_title("Permutation importance (top features)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return RankingReport(csv_path, out_path, top_k)


def combined_ranking_report(
    pfi_csv: str | Path,
    shap_csv: str | Path,
    out_dir: str | Path,
) -> Path:
    """PFI+SHAP Ensemble: усреднение нормированных рангов двух методов.

    Ранг 0 — самый важный признак; нормировка на (n-1) даёт значения в [0, 1].
    Итоговый CSV отсортирован по возрастанию combined_rank (лучшие сверху).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _norm_ranks(df: pd.DataFrame, value_col: str) -> pd.Series:
        ranks = df[value_col].rank(ascending=False, method="average") - 1
        denom = max(len(df) - 1, 1)
        return pd.Series((ranks / denom).to_numpy(), index=df["feature"].to_numpy())

    pfi_ranks = _norm_ranks(pd.read_csv(pfi_csv), "importance_mean")
    shap_ranks = _norm_ranks(pd.read_csv(shap_csv), "mean_abs_shap")

    merged = pd.concat({"pfi_rank_norm": pfi_ranks, "shap_rank_norm": shap_ranks}, axis=1)
    merged["combined_rank"] = merged.mean(axis=1, skipna=True)
    merged = merged.sort_values("combined_rank").reset_index(names="feature")

    csv_path = out_dir / "combined_ranking.csv"
    merged.to_csv(csv_path, index=False)
    log.info(f"Combined PFI+SHAP ranking for {len(merged)} features -> {csv_path.name}")
    return csv_path
