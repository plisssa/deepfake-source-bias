"""Optuna-обвязка для подбора гиперпараметров классических моделей и/или extractor-ов.

Используется и в основном CLI (если cfg.tuning.enabled=true), и отдельно
скриптом scripts/tune_pipeline.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TuneResult:
    best_value: float
    best_params: dict[str, Any]
    study: optuna.Study


def tune(
    objective: Callable[[optuna.Trial], float],
    n_trials: int = 50,
    direction: str = "maximize",
    timeout: int | None = None,
    study_name: str | None = None,
    seed: int = 42,
    storage: str | None = "sqlite:///optuna.db",
    load_if_exists: bool = True,
    enqueue: list[dict[str, Any]] | None = None,
) -> TuneResult:
    """Запуск Optuna-исследования.

    storage: URL хранилища (по умолчанию sqlite:///optuna.db относительно cwd —
    study переживает перезапуск и может быть продолжен благодаря load_if_exists).
    storage=None — чисто in-memory study (используется в тестах).
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists if storage is not None else False,
    )
    # Первыми — заданные конфигурации (обычно параметры по умолчанию). Иначе
    # «тюнинг» может оказаться хуже его отсутствия: 29.09.2026 у col2 лучший
    # из 50 trials дал на val 0.9515 при 0.9692 у параметров по умолчанию, и на
    # test тюнингованная модель проиграла (0.951 против 0.968). Умолчание лежало
    # внутри пространства, но TPE на 36 тыс. признаков к нему не пришёл.
    # enqueue_trial не дублируется при продолжении существующего study.
    if enqueue and len(study.trials) == 0:
        for params in enqueue:
            study.enqueue_trial(params)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    log.info(f"Best {direction}: {study.best_value:.4f}")
    log.info(f"Best params: {study.best_params}")
    return TuneResult(study.best_value, study.best_params, study)


#: значения библиотек по умолчанию для ключей, которые sklearn-обёртка xgboost
#: отдаёт как None («не задано — применится внутреннее умолчание»)
_LIBRARY_DEFAULTS: dict[str, float] = {
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}


def default_trial_params(model_name: str, cfg_params: dict, space_fn) -> dict[str, Any]:
    """Фактические параметры модели по умолчанию для ВСЕХ ключей пространства.

    Нужно, чтобы первая попытка Optuna была ровно той моделью, что в бенчмарке.
    Ключи, не заданные в конфиге модели, берутся из get_params() собранной модели,
    а если там None — из известных умолчаний библиотеки. Ключ без известного
    умолчания — ошибка: первая попытка иначе молча отличалась бы от бенчмарка."""
    from deepfakes.models.classical import ModelSpec, build_model

    probe = optuna.create_study()
    trial = probe.ask()
    space_fn(trial)
    keys = list(trial.params)
    est = build_model(ModelSpec(name=model_name, params=dict(cfg_params)))
    effective = est.get_params() if hasattr(est, "get_params") else {}
    out: dict[str, Any] = {}
    for k in keys:
        if k in cfg_params:
            out[k] = cfg_params[k]
        elif effective.get(k) is not None:
            out[k] = effective[k]
        elif k in _LIBRARY_DEFAULTS:
            out[k] = _LIBRARY_DEFAULTS[k]
        else:
            raise ValueError(f"нет известного умолчания для '{k}' ({model_name})")
    return out


# ---------- default search spaces ----------


def xgboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def rf_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
    }


def logreg_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
    }


def lightgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    """Имена параметров совпадают с LGBMClassifier (см. build_model в models/classical.py)."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def catboost_space(trial: optuna.Trial) -> dict[str, Any]:
    """Имена параметров совпадают с CatBoostClassifier (depth/iterations/l2_leaf_reg)."""
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 800, step=100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
    }


def svm_space(trial: optuna.Trial) -> dict[str, Any]:
    """Параметры SVC (внутри Pipeline со StandardScaler, см. build_model)."""
    return {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
    }


_MLP_HIDDEN_CHOICES = ["64", "128", "128-64", "256-128"]


def _parse_hidden_layers(s: str) -> tuple[int, ...]:
    """'128-64' -> (128, 64). Optuna categorical не хранит кортежи — кодируем строкой."""
    return tuple(int(x) for x in s.split("-"))


def mlp_space(trial: optuna.Trial) -> dict[str, Any]:
    """Параметры MLPClassifier. hidden_layer_sizes кодируется строкой в trial,
    но возвращается уже кортежем — build_model получает валидное значение."""
    hidden = trial.suggest_categorical("hidden_layer_sizes", _MLP_HIDDEN_CHOICES)
    return {
        "hidden_layer_sizes": _parse_hidden_layers(hidden),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


def normalize_model_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Приводит study.best_params к виду, который принимает build_model.

    Нужно для mlp: в best_params hidden_layer_sizes лежит строкой ('128-64'),
    а MLPClassifier ждёт кортеж.
    """
    out = dict(params)
    if model_name == "mlp" and isinstance(out.get("hidden_layer_sizes"), str):
        out["hidden_layer_sizes"] = _parse_hidden_layers(out["hidden_layer_sizes"])
    return out


MODEL_SPACES = {
    "xgboost": xgboost_space,
    "random_forest": rf_space,
    "logreg": logreg_space,
    "lightgbm": lightgbm_space,
    "catboost": catboost_space,
    "svm": svm_space,
    "mlp": mlp_space,
}


# ---------- pipeline-level search spaces (extractor / augmentation) ----------


def freq4_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """FREQ-4: подбор HF-GAN intensity, числа bins и (опц.) staged-режима.

    n_stages/lr_schedule предлагаются ТОЛЬКО при staged_mode != 'none' —
    иначе мёртвые размерности засоряют TPE.
    """
    params = {
        "gan_intensity": trial.suggest_float("gan_intensity", 0.005, 0.5, log=True),
        "n_bins": trial.suggest_int("n_bins", 4, 16),
        "use_gan_augment": True,
        "staged_mode": trial.suggest_categorical("staged_mode", ["none", "continual"]),
    }
    if params["staged_mode"] != "none":
        params["n_stages"] = trial.suggest_int("n_stages", 2, 5)
        params["lr_schedule"] = trial.suggest_categorical(
            "lr_schedule", ["warmup_cosine", "step_decay"]
        )
    return params


def col2_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """COL-2: размер патчей + цветовое пространство."""
    patch_sizes_options = ["[8]", "[16]", "[32]", "[8,16]", "[8,16,32]", "[16,32]"]
    return {
        "patch_sizes_str": trial.suggest_categorical("patch_sizes_str", patch_sizes_options),
        "colorspace": trial.suggest_categorical("colorspace", ["RGB", "HSV", "YCbCr", "Lab"]),
    }


def tex2_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """TEX-2: радиусы, метод LBP, размер сетки блоков."""
    radii_options = ["[1]", "[1,2]", "[1,2,3]", "[2,3]"]
    return {
        "radii_str": trial.suggest_categorical("radii_str", radii_options),
        "method": trial.suggest_categorical("method", ["uniform", "default", "simple"]),
        "block_grid": trial.suggest_categorical("block_grid", [None, 2, 4]),
    }


def freq3_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """FREQ-3: выбор материнского вейвлета и глубины разложения.

    Имена параметров совпадают с ключами configs/pipeline/freq3.yaml
    (и Freq3Pipeline.from_cfg): wavelet, level.
    """
    return {
        "wavelet": trial.suggest_categorical(
            "wavelet", ["haar", "db2", "db4", "sym4", "coif2", "bior2.2"]
        ),
        "level": trial.suggest_int("level", 2, 4),
    }


def noise1_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "denoise_strength": trial.suggest_float("denoise_strength", 3.0, 25.0),
        "patch_grid": trial.suggest_categorical("patch_grid", [1, 2, 4]),
    }


def grad4_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """GRAD-4 (LoG): sigma и размер патча.

    Имена совпадают с ключами Grad4Pipeline.from_cfg (sigma, patch_size).
    подбор был бы молча проигнорирован.
    """
    return {
        "sigma": trial.suggest_float("sigma", 0.6, 2.5),
        "zc_threshold": trial.suggest_float("zc_threshold", 0.1, 1.0),
        "patch_size": trial.suggest_categorical("patch_size", [8, 16, 32]),
    }


def tex5_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """TEX-5 (Gabor bank): набор длин волн + сетка патчей.

    Имена совпадают с ключами Tex5Pipeline.from_cfg (wavelengths, patch_grid);
    списки кодируются строкой (см. merge_pipeline_overrides).
    """
    wavelengths_options = ["[4,8]", "[4,8,16]", "[8,16]", "[2,4,8]", "[4,8,16,32]"]
    return {
        "wavelengths_str": trial.suggest_categorical("wavelengths_str", wavelengths_options),
        "patch_grid": trial.suggest_categorical("patch_grid", [1, 2, 4]),
    }


def tex6_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """TEX-6 (GLCM): число уровней квантования + набор дистанций.

    Имена совпадают с ключами Tex6Pipeline.from_cfg (levels, distances).
    """
    distances_options = ["[1]", "[1,2]", "[1,2,3]", "[2,3]"]
    return {
        "levels": trial.suggest_categorical("levels", [16, 32, 64]),
        "distances_str": trial.suggest_categorical("distances_str", distances_options),
    }


def ela_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """ELA: качество повторного JPEG-сжатия (ключ quality из ElaPipeline.from_cfg)."""
    return {
        "quality": trial.suggest_int("quality", 60, 95),
    }


def tex4_pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """TEX-4 (HOG): размер ячейки (ключ pixels_per_cell из Tex4Pipeline.from_cfg)."""
    return {
        "pixels_per_cell": trial.suggest_categorical("pixels_per_cell", [8, 16]),
    }


PIPELINE_SPACES = {
    "freq3": freq3_pipeline_space,
    "freq4": freq4_pipeline_space,
    "col2": col2_pipeline_space,
    "tex2": tex2_pipeline_space,
    "noise1": noise1_pipeline_space,
    "grad4": grad4_pipeline_space,
    "tex5": tex5_pipeline_space,
    "tex6": tex6_pipeline_space,
    "ela": ela_pipeline_space,
    "tex4": tex4_pipeline_space,
}


def merge_pipeline_overrides(base_cfg: dict, suggested: dict) -> dict:
    """Аккуратное вмешательство в pipeline-конфиг: преобразование строк-списков
    обратно в списки int (Optuna не умеет suggest со списками)."""
    out = dict(base_cfg)
    for k, v in suggested.items():
        if k.endswith("_str") and isinstance(v, str):
            real_key = k[:-4]
            out[real_key] = [int(x) for x in v.strip("[]").split(",") if x]
        else:
            out[k] = v
    return out
