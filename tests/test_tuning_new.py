"""Тесты нового Optuna-тюнинга: новые MODEL_SPACES, новые PIPELINE_SPACES,
условные suggest для FREQ-4 и in-memory study (storage=None).
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import optuna
import pytest
import yaml

from deepfakes.models.classical import ModelSpec, build_model
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tuning.optuna_runner import (
    MODEL_SPACES,
    PIPELINE_SPACES,
    merge_pipeline_overrides,
    normalize_model_params,
    tune,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

NEW_MODELS = ["lightgbm", "catboost", "svm", "mlp"]
NEW_PIPELINES = ["grad4", "tex5", "tex6", "ela", "tex4"]

CONFIGS_DIR = "configs/pipeline"


@pytest.fixture(scope="module")
def tiny_xy() -> tuple[np.ndarray, np.ndarray]:
    """Крошечные синтетические табличные данные: 60 объектов, 8 признаков."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=60) > 0).astype(int)
    return X, y


# ---------- новые MODEL_SPACES ----------


@pytest.mark.parametrize("model_name", NEW_MODELS)
def test_new_model_space_three_trials(model_name, tiny_xy):
    """3 трайла на крошечных данных проходят; best_params принимаются build_model."""
    X, y = tiny_xy
    space_fn = MODEL_SPACES[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        # быстрее: урезаем тяжёлые параметры до минимума на тестовых данных
        for k in ("n_estimators", "iterations"):
            if k in params:
                params[k] = 10
        model = build_model(ModelSpec(name=model_name, params=params))
        model.fit(X, y)
        return float(model.score(X, y))

    result = tune(objective, n_trials=3, storage=None, study_name=None, seed=0)
    assert len(result.study.trials) == 3
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in result.study.trials)

    # best_params (после нормализации, см. mlp) принимаются фабрикой — модель строится
    params = normalize_model_params(model_name, result.best_params)
    for k in ("n_estimators", "iterations"):
        if k in params:
            params[k] = 10
    model = build_model(ModelSpec(name=model_name, params=params))
    model.fit(X, y)
    assert model.predict(X).shape == (len(y),)


def test_mlp_hidden_layer_sizes_normalized():
    """Строка '128-64' из best_params превращается в кортеж (128, 64)."""
    params = normalize_model_params("mlp", {"hidden_layer_sizes": "128-64", "alpha": 1e-4})
    assert params["hidden_layer_sizes"] == (128, 64)


# ---------- новые PIPELINE_SPACES ----------


def _from_cfg_keys(pipeline_name: str) -> set[str]:
    """Ключи pipeline_cfg.get(...) из исходника from_cfg соответствующего класса."""
    cls = get_pipeline_class(pipeline_name)
    src = inspect.getsource(cls.from_cfg)
    return set(re.findall(r'pipeline_cfg\.get\(\s*"([^"]+)"', src))


@pytest.mark.parametrize("pipeline_name", NEW_PIPELINES)
def test_new_pipeline_space_keys_subset_of_from_cfg(pipeline_name):
    """Ключи пространства (без суффикса _str) ⊆ ключей from_cfg пайплайна."""
    space_fn = PIPELINE_SPACES[pipeline_name]
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: (space_fn(t), 0.0)[1], n_trials=1)
    suggested = space_fn(study.best_trial)
    real_keys = {k.removesuffix("_str") for k in suggested}
    cfg_keys = _from_cfg_keys(pipeline_name)
    assert (
        real_keys <= cfg_keys
    ), f"{pipeline_name}: keys {real_keys - cfg_keys} not read by from_cfg ({cfg_keys})"


@pytest.mark.parametrize("pipeline_name", NEW_PIPELINES)
def test_new_pipeline_constructs_from_merged_cfg(pipeline_name):
    """from_cfg с base-конфигом + best_params реально конструирует пайплайн."""
    space_fn = PIPELINE_SPACES[pipeline_name]
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(lambda t: (space_fn(t), 0.0)[1], n_trials=1)

    with open(f"{CONFIGS_DIR}/{pipeline_name}.yaml") as f:
        base_cfg = yaml.safe_load(f)
    merged = merge_pipeline_overrides(base_cfg, study.best_params)

    cls = get_pipeline_class(pipeline_name)
    pipe = cls.from_cfg(merged, ModelSpec(name="logreg", params={}), cache_dir=None)
    assert pipe is not None


# ---------- условные suggest FREQ-4 ----------


def test_freq4_no_staged_params_when_mode_none():
    """При staged_mode='none' n_stages/lr_schedule не предлагаются вовсе."""
    space_fn = PIPELINE_SPACES["freq4"]
    trial = optuna.trial.FixedTrial({"gan_intensity": 0.02, "n_bins": 8, "staged_mode": "none"})
    params = space_fn(trial)
    assert "n_stages" not in params
    assert "lr_schedule" not in params


def test_freq4_staged_params_when_mode_continual():
    space_fn = PIPELINE_SPACES["freq4"]
    trial = optuna.trial.FixedTrial(
        {
            "gan_intensity": 0.02,
            "n_bins": 8,
            "staged_mode": "continual",
            "n_stages": 3,
            "lr_schedule": "step_decay",
        }
    )
    params = space_fn(trial)
    assert params["n_stages"] == 3
    assert params["lr_schedule"] == "step_decay"


# ---------- storage=None -> in-memory study ----------


def test_tune_with_storage_none_runs_in_memory():
    """tune(..., storage=None) работает без sqlite-файла."""
    result = tune(
        lambda t: t.suggest_float("x", -1.0, 1.0) ** 2,
        n_trials=3,
        direction="minimize",
        storage=None,
        study_name="inmem_test",
        seed=0,
    )
    assert len(result.study.trials) == 3
    assert result.best_value >= 0.0
