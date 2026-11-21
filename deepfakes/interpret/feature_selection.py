"""INT-1 & INT-8: корреляция, mutual information, ANOVA F-тест и Gini importance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


def default_feature_names(n_features: int) -> list[str]:
    """Имена-заглушки вида f000, f001, ... — когда экстрактор не отдал имён."""
    width = max(3, len(str(max(n_features - 1, 0))))
    return [f"f{i:0{width}d}" for i in range(n_features)]


@dataclass
class RankingReport:
    """Единый контракт отчётов важности: полный CSV + bar-plot топ-K."""

    importance_csv: Path
    plot_path: Path
    top_k: int


def _save_ranking_report(
    df: pd.DataFrame,
    value_col: str,
    out_dir: Path,
    csv_name: str,
    plot_name: str,
    xlabel: str,
    title: str,
    top_k: int,
) -> RankingReport:
    """Сохранить отсортированный df в CSV и нарисовать barh по топ-K строк."""
    csv_path = out_dir / csv_name
    df.to_csv(csv_path, index=False)

    top = df.head(top_k)
    fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.2)))
    ax.barh(top["feature"][::-1], top[value_col][::-1])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    plot_path = out_dir / plot_name
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return RankingReport(csv_path, plot_path, top_k)


@dataclass
class CorrelationReport:
    corr_matrix_path: Path
    redundant_pairs_csv: Path
    n_redundant: int


def correlation_matrix_report(
    X: np.ndarray,
    feature_names: list[str] | None,
    out_dir: str | Path,
    threshold: float = 0.95,
    max_features_in_plot: int = 80,
) -> CorrelationReport:
    """INT-1: heatmap корреляций + список redundant пар (|r| >= threshold).

    Если фич больше max_features_in_plot — heatmap строится по топ-K с наибольшей
    дисперсией; список пар при этом строится по всему набору.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_feat = X.shape[1]
    feature_names = feature_names or [f"f{i}" for i in range(n_feat)]
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr().fillna(0)

    # Redundant pairs (всегда по полной матрице)
    pairs: list[tuple[str, str, float]] = []
    mask_upper = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
    abs_corr = np.abs(corr.values) * mask_upper
    rows, cols = np.where(abs_corr >= threshold)
    for r, c in zip(rows, cols):
        pairs.append((feature_names[r], feature_names[c], float(corr.values[r, c])))
    pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr"])
    pairs_csv = out_dir / "correlation_redundant_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)

    # Heatmap (по топ-K дисперсии для читаемости)
    plot_cols = feature_names
    plot_df = df
    if n_feat > max_features_in_plot:
        top_idx = np.argsort(df.var().values)[::-1][:max_features_in_plot]
        plot_cols = [feature_names[i] for i in top_idx]
        plot_df = df[plot_cols]
    plot_corr = plot_df.corr().fillna(0)
    fig, ax = plt.subplots(figsize=(max(8, len(plot_cols) * 0.15), max(6, len(plot_cols) * 0.13)))
    sns.heatmap(plot_corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, cbar=True)
    ax.set_title(f"Correlation matrix (top-{len(plot_cols)} by variance)")
    fig.tight_layout()
    heatmap_path = out_dir / "correlation_matrix.png"
    fig.savefig(heatmap_path, dpi=120)
    plt.close(fig)

    log.info(
        f"INT-1: {len(pairs)} redundant pairs (|r|>={threshold}); heatmap saved to {heatmap_path.name}"
    )
    return CorrelationReport(heatmap_path, pairs_csv, len(pairs))


@dataclass
class MIReport:
    importance_csv: Path
    plot_path: Path
    top_k: int


def mutual_info_report(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None,
    out_dir: str | Path,
    top_k: int = 30,
    seed: int = 42,
) -> MIReport:
    """INT-8: mutual_info_classif → top-K bar chart + CSV полный."""
    from sklearn.feature_selection import mutual_info_classif

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    mi = mutual_info_classif(X, y, random_state=seed, discrete_features=False)
    df = pd.DataFrame({"feature": feature_names, "mi": mi}).sort_values("mi", ascending=False)
    csv_path = out_dir / "mutual_information.csv"
    df.to_csv(csv_path, index=False)

    top = df.head(top_k)
    fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.2)))
    ax.barh(top["feature"][::-1], top["mi"][::-1])
    ax.set_xlabel("Mutual information")
    ax.set_title(f"INT-8: Top-{top_k} features by MI")
    fig.tight_layout()
    plot_path = out_dir / "mutual_information_top.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    log.info(f"INT-8: MI computed for {X.shape[1]} features; top-{top_k} plotted")
    return MIReport(csv_path, plot_path, top_k)


def anova_report(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None,
    out_dir: str | Path,
    top_k: int = 30,
) -> RankingReport:
    """ANOVA F-тест (f_classif): CSV (feature, F, p) + bar-plot топ-K по F.

    NaN в F (константные признаки) заменяются нулями, чтобы сортировка и
    plot не падали.
    """
    from sklearn.feature_selection import f_classif

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = feature_names or default_feature_names(X.shape[1])

    f_stat, p_val = f_classif(X, y)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "F": np.nan_to_num(f_stat, nan=0.0),
            "p": np.nan_to_num(p_val, nan=1.0),
        }
    ).sort_values("F", ascending=False)
    rep = _save_ranking_report(
        df,
        value_col="F",
        out_dir=out_dir,
        csv_name="anova_f.csv",
        plot_name="anova_f_top.png",
        xlabel="ANOVA F-statistic",
        title=f"Top-{top_k} features by ANOVA F",
        top_k=top_k,
    )
    log.info(f"ANOVA: F computed for {X.shape[1]} features; top-{top_k} plotted")
    return rep


def gini_report(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None,
    out_dir: str | Path,
    top_k: int = 30,
    seed: int = 42,
) -> RankingReport:
    """Gini importance: RandomForest(200) → feature_importances_ → CSV + plot."""
    from sklearn.ensemble import RandomForestClassifier

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = feature_names or default_feature_names(X.shape[1])

    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    df = pd.DataFrame(
        {"feature": feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    rep = _save_ranking_report(
        df,
        value_col="importance",
        out_dir=out_dir,
        csv_name="gini_importance.csv",
        plot_name="gini_importance_top.png",
        xlabel="Gini importance (RandomForest)",
        title=f"Top-{top_k} features by Gini importance",
        top_k=top_k,
    )
    log.info(f"Gini: importances computed for {X.shape[1]} features; top-{top_k} plotted")
    return rep
