"""Статистика сравнения детекторов: проверки на свойствах, которые обязаны выполняться."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from deepfakes.eval.stats import (
    _midrank,
    bootstrap_auc_ci,
    delong_test,
    friedman_nemenyi,
    holm,
    paired_bootstrap_diff,
)


def _data(n=1500, shift1=1.0, shift2=0.6, rho=0.5, seed=0):
    """Два коррелированных скора на одних объектах, AUC разные."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    common = rng.normal(size=n)
    e1 = rho * common + np.sqrt(1 - rho**2) * rng.normal(size=n)
    e2 = rho * common + np.sqrt(1 - rho**2) * rng.normal(size=n)
    return y, e1 + shift1 * y, e2 + shift2 * y


def test_midrank_handles_ties():
    np.testing.assert_allclose(_midrank(np.array([3.0, 1.0, 3.0, 2.0])), [3.5, 1.0, 3.5, 2.0])


def test_delong_auc_matches_sklearn():
    y, s1, s2 = _data()
    r = delong_test(y, s1, s2)
    assert r.auc1 == pytest.approx(roc_auc_score(y, s1), abs=1e-12)
    assert r.auc2 == pytest.approx(roc_auc_score(y, s2), abs=1e-12)


def test_delong_identical_scores_is_not_significant():
    y, s1, _ = _data()
    r = delong_test(y, s1, s1)
    assert r.diff == 0 and r.p_value == 1.0


def test_delong_detects_real_difference():
    y, s1, s2 = _data(n=3000, shift1=1.0, shift2=0.5)
    assert delong_test(y, s1, s2).p_value < 1e-4


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_delong_false_positive_rate_is_calibrated(rho):
    """Под H0 (одинаковая сила) доля p<0.05 обязана быть около 5%.

    Проверка на ОДНОМ seed для статистической процедуры бессмысленна: первая
    версия этого теста взяла seed, оказавшийся на 0.3-м процентиле нулевого
    распределения (p=0.004), и «поймала» несуществующую ошибку. Калибровку
    проверяют частотой ложных срабатываний на многих повторах."""
    p = np.array(
        [
            delong_test(*_data(n=600, shift1=0.8, shift2=0.8, rho=rho, seed=s)).p_value
            for s in range(300)
        ]
    )
    rate = float(np.mean(p < 0.05))
    assert 0.02 <= rate <= 0.09, f"доля ложных срабатываний {rate:.3f} при α=0.05"


def test_delong_std_agrees_with_paired_bootstrap():
    """Стандартная ошибка разности по DeLong ≈ по парному bootstrap."""
    y, s1, s2 = _data(n=2000, seed=3)
    r = delong_test(y, s1, s2)
    se_delong = abs(r.diff / r.z)
    rng = np.random.default_rng(1)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(1500):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        d.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    assert se_delong == pytest.approx(np.std(d), rel=0.15)


def test_delong_is_symmetric():
    y, s1, s2 = _data()
    a, b = delong_test(y, s1, s2), delong_test(y, s2, s1)
    assert a.diff == pytest.approx(-b.diff) and a.p_value == pytest.approx(b.p_value)


def test_bootstrap_ci_contains_point_and_narrows_with_n():
    y, s, _ = _data(n=400)
    auc, lo, hi = bootstrap_auc_ci(y, s, n_boot=500)
    assert lo < auc < hi
    y2, s2, _ = _data(n=4000)
    _, lo2, hi2 = bootstrap_auc_ci(y2, s2, n_boot=500)
    assert (hi2 - lo2) < (hi - lo)


def test_paired_bootstrap_diff_sign():
    y, s1, s2 = _data(n=2000)
    d, lo, _hi = paired_bootstrap_diff(y, s1, s2, n_boot=500)
    assert d > 0 and lo > 0, "лучший скор должен быть значимо лучше"


def test_holm_known_values():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.all(holm([0.5, 0.9]) <= 1.0)


def test_friedman_ranks_and_cd():
    # 3 метода, 10 блоков: метод 0 всегда лучший, 2 — худший
    rng = np.random.default_rng(0)
    base = np.array([0.9, 0.8, 0.7])
    scores = base + rng.normal(0, 0.005, (10, 3))
    r = friedman_nemenyi(scores, ["a", "b", "c"])
    np.testing.assert_allclose(r.mean_ranks, [1, 2, 3])
    assert r.p_value < 0.001
    # табличное значение Demšar 2006: k=3, α=0.05 → q=2.343
    assert r.critical_difference == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 10)), rel=1e-3)
