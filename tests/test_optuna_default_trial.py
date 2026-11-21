"""Первая попытка Optuna — ровно модель бенчмарка, поэтому тюнинг не хуже умолчания.

Регрессия 29.09.2026: у col2 лучший из 50 trials дал на val 0.9515 при 0.9692 у
параметров по умолчанию; на test тюнинг проиграл (0.951 против 0.968)."""

from __future__ import annotations

import optuna
import pytest

from deepfakes.tuning.optuna_runner import default_trial_params, tune, xgboost_space

XGB_CFG = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def test_default_trial_covers_every_space_key_with_benchmark_values():
    d = default_trial_params("xgboost", XGB_CFG, xgboost_space)
    probe = optuna.create_study().ask()
    xgboost_space(probe)
    assert set(d) == set(probe.params), "каждый ключ пространства должен быть задан"
    for k, v in XGB_CFG.items():
        assert d[k] == v
    # не заданные в конфиге — это умолчания xgboost, а не сэмпл
    assert d["min_child_weight"] == 1.0 and d["reg_lambda"] == 1.0


def test_first_trial_is_exactly_the_default():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = default_trial_params("xgboost", XGB_CFG, xgboost_space)
    seen = []

    def obj(t):
        seen.append(xgboost_space(t))
        return 0.0

    tune(obj, n_trials=3, storage=None, enqueue=[d])
    assert seen[0] == d


def test_tuning_cannot_lose_to_default_on_objective():
    """Если умолчание — оптимум, лучший результат study обязан быть ≥ его."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = default_trial_params("xgboost", XGB_CFG, xgboost_space)

    def obj(t):
        p = xgboost_space(t)
        return -abs(p["learning_rate"] - 0.05) - abs(p["max_depth"] - 6)  # максимум у умолчания

    res = tune(obj, n_trials=5, storage=None, enqueue=[d])
    assert res.best_value == pytest.approx(0.0)
