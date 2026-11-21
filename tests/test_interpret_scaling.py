"""Предотбор признаков для дорогих методов интерпретации.

Регрессия 27.09.2026: у col2 вектор 36 288 признаков, и `correlation_matrix_report`
строил матрицу 36288² (10.5 ГБ в float64, плюс маска и произведение того же
размера — пик под 30 ГБ). Стадия interpret не завершалась ни за 4 часа, ни за 24;
поднимать лимит было бессмысленно, нужен другой масштаб метода.
"""

from __future__ import annotations

import numpy as np

from scripts.run_interpret import (
    DEFAULT_MAX_FEATURES,
    EXPENSIVE_KINDS,
    VALID_KINDS,
    select_topk_by_anova,
)


def test_expensive_kinds_are_the_superlinear_ones():
    """Состав дорогих методов должен быть подмножеством известных."""
    assert EXPENSIVE_KINDS <= VALID_KINDS
    assert {"correlation", "mutual_info", "pfi"} == EXPENSIVE_KINDS
    # anova/gini/shap масштабируются и предотбора не требуют
    assert not EXPENSIVE_KINDS & {"anova", "gini", "shap"}


def test_selects_informative_features():
    """Предотбор обязан находить признаки, реально связанные с метками."""
    rng = np.random.default_rng(0)
    n, d = 400, 60
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, n)
    informative = [3, 17, 42]
    for j in informative:
        X[:, j] += y * 3.0  # явный сдвиг по классу

    idx, f_vals = select_topk_by_anova(X, y, k=5)
    assert idx is not None
    assert len(idx) == 5
    assert set(informative) <= set(idx.tolist()), "информативные признаки не отобраны"
    assert np.all(f_vals >= 0)


def test_index_order_is_ascending_for_readable_names():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 40))
    y = rng.integers(0, 2, 200)
    idx, _ = select_topk_by_anova(X, y, k=10)
    assert list(idx) == sorted(idx), "порядок индексов должен сохраняться"


def test_k_larger_than_d_returns_all():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 7))
    y = rng.integers(0, 2, 100)
    idx, _ = select_topk_by_anova(X, y, k=999)
    assert len(idx) == 7


def test_constant_features_do_not_crash():
    """f_classif даёт NaN на константных признаках — их надо гасить, не падать."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 10))
    X[:, 4] = 5.0  # константа
    X[:, 7] = 0.0
    y = rng.integers(0, 2, 120)
    idx, f_vals = select_topk_by_anova(X, y, k=6)
    assert idx is not None and len(idx) == 6
    assert np.all(np.isfinite(f_vals))


def test_no_labels_means_no_selection():
    """Без меток предотбор невозможен — функция обязана сказать это явно."""
    X = np.zeros((10, 5))
    idx, f_vals = select_topk_by_anova(X, None, k=3)
    assert idx is None and f_vals is None


def test_default_threshold_is_below_problem_size():
    """Порог должен реально спасать от случая, который всё сломал (36288)."""
    assert DEFAULT_MAX_FEATURES < 36288
    corr_gb = (DEFAULT_MAX_FEATURES**2 * 8) / 1024**3
    assert corr_gb < 0.1, f"матрица корреляций всё ещё {corr_gb:.2f} ГБ"
