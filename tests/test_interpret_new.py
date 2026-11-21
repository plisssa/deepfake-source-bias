"""Тесты новых методов важности признаков: ANOVA, Gini, PFI, SHAP, combined ranking."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from deepfakes.interpret.explain import (
    combined_ranking_report,
    permutation_importance_plot,
    shap_summary,
)
from deepfakes.interpret.feature_selection import (
    anova_report,
    default_feature_names,
    gini_report,
)

N_FEATURES = 20


@pytest.fixture(scope="module")
def synth() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Синтетика 200x20 с 5 информативными признаками."""
    X, y = make_classification(
        n_samples=200, n_features=N_FEATURES, n_informative=5, random_state=0
    )
    return X, y, default_feature_names(N_FEATURES)


@pytest.fixture(scope="module")
def rf_model(synth) -> RandomForestClassifier:
    """Обученный RandomForest — модельный «пайплайн» для SHAP/PFI без BasePipeline."""
    X, y, _ = synth
    return RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)


def test_default_feature_names():
    names = default_feature_names(12)
    assert len(names) == 12
    assert names[0] == "f000" and names[11] == "f011"
    assert len(default_feature_names(2000)) == 2000  # ширина растёт при необходимости


def test_anova_report(synth, tmp_path):
    X, y, names = synth
    rep = anova_report(X, y, names, tmp_path, top_k=10)
    assert rep.importance_csv.exists() and rep.plot_path.exists()
    df = pd.read_csv(rep.importance_csv)
    assert set(df.columns) == {"feature", "F", "p"}
    assert len(df) == N_FEATURES
    assert (df["F"] >= 0).all()
    assert df["p"].between(0, 1).all()
    # CSV отсортирован по убыванию F
    assert df["F"].is_monotonic_decreasing


def test_anova_none_names(synth, tmp_path):
    X, y, _ = synth
    rep = anova_report(X, y, None, tmp_path, top_k=5)
    df = pd.read_csv(rep.importance_csv)
    assert set(df["feature"]) == set(default_feature_names(N_FEATURES))


def test_gini_report(synth, tmp_path):
    X, y, names = synth
    rep = gini_report(X, y, names, tmp_path, top_k=10, seed=0)
    assert rep.importance_csv.exists() and rep.plot_path.exists()
    df = pd.read_csv(rep.importance_csv)
    assert set(df.columns) == {"feature", "importance"}
    assert len(df) == N_FEATURES
    assert (df["importance"] >= 0).all()
    assert df["importance"].sum() == pytest.approx(1.0, abs=1e-6)


def test_pfi_report(synth, rf_model, tmp_path):
    X, y, names = synth
    rep = permutation_importance_plot(
        rf_model, X, y, tmp_path / "pfi.png", n_repeats=3, feature_names=names, top_k=10
    )
    assert rep is not None
    assert rep.importance_csv.exists() and rep.plot_path.exists()
    df = pd.read_csv(rep.importance_csv)
    assert {"feature", "importance_mean", "importance_std"} <= set(df.columns)
    assert len(df) == N_FEATURES


def test_shap_report_new_api(synth, rf_model, tmp_path):
    """RandomForest + новый API SHAP (3D ndarray) не должны падать/скипаться."""
    X, _, names = synth
    rep = shap_summary(rf_model, X, tmp_path / "shap.png", n_samples=60, feature_names=names)
    assert rep is not None, "shap_summary вернул None — SHAP упал или был пропущен"
    assert rep.importance_csv.exists() and rep.plot_path.exists()
    df = pd.read_csv(rep.importance_csv)
    assert set(df.columns) == {"feature", "mean_abs_shap"}
    assert len(df) == N_FEATURES
    assert (df["mean_abs_shap"] >= 0).all()
    assert df["mean_abs_shap"].sum() > 0


def test_combined_ranking(synth, rf_model, tmp_path):
    """Combined ranking (PFI+SHAP Ensemble) содержит все признаки, ранги в [0, 1]."""
    X, y, names = synth
    pfi_rep = permutation_importance_plot(
        rf_model, X, y, tmp_path / "pfi.png", n_repeats=3, feature_names=names
    )
    shap_rep = shap_summary(rf_model, X, tmp_path / "shap.png", n_samples=60, feature_names=names)
    assert pfi_rep is not None and shap_rep is not None

    csv_path = combined_ranking_report(pfi_rep.importance_csv, shap_rep.importance_csv, tmp_path)
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert set(df["feature"]) == set(names)
    assert len(df) == N_FEATURES
    for col in ("pfi_rank_norm", "shap_rank_norm", "combined_rank"):
        assert df[col].between(0, 1).all()
    assert df["combined_rank"].is_monotonic_increasing


def test_parse_kinds():
    from scripts.run_interpret import _parse_kinds

    assert _parse_kinds("both") == {"correlation", "mutual_info"}
    assert _parse_kinds("all") == {"correlation", "mutual_info", "shap", "pfi", "anova", "gini"}
    assert _parse_kinds("anova") == {"anova"}
    with pytest.raises(ValueError):
        _parse_kinds("nope")
