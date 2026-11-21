"""Bootstrap confidence intervals, DeLong comparisons and Holm correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------- bootstrap


def bootstrap_auc_ci(
    y: np.ndarray, s: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(AUC, нижняя, верхняя граница) — перцентильный стратифицированный bootstrap."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        aucs[b] = roc_auc_score(y[idx], s[idx])
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def paired_bootstrap_diff(
    y, s1, s2, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """CI для разности AUC1 − AUC2 на одних объектах (одни и те же выборки)."""
    y = np.asarray(y).astype(int)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        d[b] = roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx])
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(roc_auc_score(y, s1) - roc_auc_score(y, s2)), float(lo), float(hi)


# ------------------------------------------------------------------- DeLong


def _midrank(x: np.ndarray) -> np.ndarray:
    """Средние ранги (1-based) с усреднением связок."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks_sorted
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014). preds_sorted: (k, n), первые m столбцов — положительные."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds_sorted[r, :m])
        ty[r] = _midrank(preds_sorted[r, m:])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


@dataclass
class DeLongResult:
    auc1: float
    auc2: float
    diff: float
    z: float
    p_value: float


def delong_test(y, s1, s2) -> DeLongResult:
    """Двусторонний тест DeLong для двух скоров на ОДНИХ И ТЕХ ЖЕ объектах."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")  # положительные вперёд
    m = int(y.sum())
    preds = np.vstack([np.asarray(s1, float), np.asarray(s2, float)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0 or not np.isfinite(var):
        z, p = 0.0, 1.0  # одинаковые (или вырожденные) скоры — различия нет
    else:
        z = diff / np.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
    return DeLongResult(float(aucs[0]), float(aucs[1]), diff, float(z), p)


def holm(pvals) -> np.ndarray:
    """Поправка Холма–Бонферрони (скорректированные p, монотонные)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------- Friedman / Nemenyi


@dataclass
class FriedmanResult:
    methods: list[str]
    mean_ranks: np.ndarray  # 1 = лучший
    statistic: float
    p_value: float
    critical_difference: float
    n_blocks: int


def friedman_nemenyi(scores: np.ndarray, methods: list[str], alpha: float = 0.05) -> FriedmanResult:
    """scores: (блоки × методы), больше = лучше. Средние ранги + CD Nemenyi.

    CD = q_α · sqrt(k(k+1) / (6N)), где q_α — квантиль стьюдентизированного
    размаха для k групп и бесконечных степеней свободы, делённый на √2
    (Demšar 2006, табл. 5)."""
    scores = np.asarray(scores, float)
    n_blocks, k = scores.shape
    ranks = np.vstack([stats.rankdata(-row) for row in scores])  # 1 — лучший
    mean_ranks = ranks.mean(axis=0)
    if n_blocks >= 2 and k >= 3:
        stat, p = stats.friedmanchisquare(*scores.T)
    else:
        stat, p = float("nan"), float("nan")
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))
    return FriedmanResult(list(methods), mean_ranks, float(stat), float(p), cd, n_blocks)
