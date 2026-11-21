"""Подбор гиперпараметров через Optuna: model и/или extractor.

Примеры:
    # Только model params (XGBoost)
    python -m scripts.tune_pipeline pipeline=col2 model=xgboost tuning=optuna_50
    # Только pipeline params (gan_intensity, n_bins, staged_mode для FREQ-4)
    python -m scripts.tune_pipeline pipeline=freq4 model=xgboost tuning=optuna_50 +tune.scope=pipeline
    # И то и другое
    python -m scripts.tune_pipeline pipeline=freq4 model=xgboost tuning=optuna_50 +tune.scope=both
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from deepfakes.eval.metrics import compute_report
from deepfakes.io.splits import load_split_csv
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run, pipeline_variant
from deepfakes.tuning.optuna_runner import (
    MODEL_SPACES,
    PIPELINE_SPACES,
    default_trial_params,
    merge_pipeline_overrides,
    normalize_model_params,
    tune,
)
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    seed_everything(cfg.seed)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_tune")

    if not cfg.tuning.enabled:
        log.warning("tuning.enabled=false. Use tuning=optuna_50 to enable.")
        return

    scope = str(cfg.get("tune", {}).get("scope", "model")).lower()
    if scope not in ("model", "pipeline", "both"):
        raise ValueError(f"tune.scope must be model|pipeline|both, got {scope}")

    metric = cfg.tuning.get("metric", "val_roc_auc")
    if metric != "val_roc_auc":
        raise ValueError(
            f"tuning.metric={metric!r} is not supported: objective is hardcoded to "
            "ROC-AUC on the validation split. Set tuning.metric to 'val_roc_auc' "
            "(or remove the field) — other metrics are not implemented yet."
        )
    storage = cfg.tuning.get("storage", "sqlite:///optuna.db")

    splits_dir = Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version
    splits = load_split_csv(splits_dir)
    pipeline_cls = get_pipeline_class(cfg.pipeline.name)
    model_space_fn = MODEL_SPACES.get(cfg.model.name)
    pipe_space_fn = PIPELINE_SPACES.get(cfg.pipeline.name)
    if scope in ("model", "both") and model_space_fn is None:
        log.warning(f"No model space for {cfg.model.name}; skipping model params")
    if scope in ("pipeline", "both") and pipe_space_fn is None:
        log.warning(f"No pipeline space for {cfg.pipeline.name}; skipping pipeline params")

    base_pipe_cfg = OmegaConf.to_container(cfg.pipeline, resolve=True)

    def objective(trial):
        # Model params
        m_params = dict(cfg.model.get("params", {}))
        if scope in ("model", "both") and model_space_fn is not None:
            m_params.update(model_space_fn(trial))
        spec = ModelSpec(name=cfg.model.name, params=m_params)
        # Pipeline params
        pipe_cfg = base_pipe_cfg
        if scope in ("pipeline", "both") and pipe_space_fn is not None:
            suggested = pipe_space_fn(trial)
            pipe_cfg = merge_pipeline_overrides(base_pipe_cfg, suggested)
        pipe = pipeline_cls.from_cfg(pipe_cfg, spec, cache_dir=cfg.paths.cache_dir)
        pipe.fit(splits["train"]["X"], splits["train"]["y"])
        scores = pipe.predict_proba(splits["val"]["X"])
        return compute_report(splits["val"]["y"], scores).roc_auc

    # Первая попытка — ТОЧНО та модель, что в бенчмарке (по всем ключам
    # пространства), иначе тюнинг может проиграть умолчанию (col2, 29.09.2026).
    default_trials = None
    if scope in ("model", "both") and model_space_fn is not None:
        default_trials = [
            default_trial_params(cfg.model.name, dict(cfg.model.get("params", {})), model_space_fn)
        ]
        log.info(f"Первая попытка Optuna — параметры по умолчанию: {default_trials[0]}")

    run_name = f"tune__{cfg.data.name}__{pipeline_variant(cfg)}__{cfg.model.name}__{scope}"
    with mlflow_run(
        run_name=run_name,
        tags={
            "kind": "tune",
            "pipeline": pipeline_variant(cfg),
            "model": cfg.model.name,
            "tune_scope": scope,
            "data": cfg.data.name,
            # протокол 2 (29.09.2026): первая попытка — ровно модель бенчмарка
            "tune_protocol": "2",
        },
    ):
        result = tune(
            objective,
            n_trials=cfg.tuning.n_trials,
            direction=cfg.tuning.direction,
            timeout=cfg.tuning.timeout_sec,
            study_name=run_name,
            seed=cfg.seed,
            storage=storage,
            load_if_exists=True,
            enqueue=default_trials,
        )
        best_params = normalize_model_params(cfg.model.name, result.best_params)
        log.info(f"Tuning finished: best {metric}={result.best_value:.4f}")
        log.info(f"Best params ({run_name}): {best_params}")
        mlflow.log_metric("best_val_auc", result.best_value)
        mlflow.set_tag("best_params", str(best_params))
        for k, v in best_params.items():
            mlflow.log_param(f"best.{k}", v)


if __name__ == "__main__":
    main()
