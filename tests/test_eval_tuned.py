"""Переоценка Optuna на test: разбор параметров и изоляция от бенчмарка."""

from __future__ import annotations

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from scripts.eval_tuned import overrides_from_best_params, tuned_studies


def test_overrides_keep_full_float_precision():
    ov = overrides_from_best_params({"learning_rate": 0.008468008575248327, "max_depth": 10})
    assert "++model.params.learning_rate=0.008468008575248327" in ov
    assert "++model.params.max_depth=10" in ov


def test_overrides_compose_with_hydra():
    """Переопределения должны реально собираться Hydra, включая новые ключи."""
    import os

    from hydra import compose, initialize_config_dir

    ov = overrides_from_best_params({"learning_rate": 0.0123, "reg_lambda": 4.9, "max_depth": 9})
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["model=xgboost", *ov])
    assert cfg.model.params.learning_rate == 0.0123
    assert cfg.model.params.reg_lambda == 4.9
    assert cfg.model.params.max_depth == 9


@pytest.fixture()
def client(tmp_path):
    uri = f"sqlite:///{tmp_path / 'm.db'}"
    mlflow.set_tracking_uri(uri)
    yield MlflowClient(tracking_uri=uri)
    mlflow.set_tracking_uri(None)


def test_tuned_studies_filters_by_domain_status_and_protocol(client):
    mlflow.set_experiment("deepfake_source_bias_tune")
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "pipeline": "col2",
                "model": "xgboost",
                "data": "gan_only",
                "best_params": "{'max_depth': 10}",
                "tune_protocol": "2",
            }
        )
        mlflow.log_metric("best_val_auc", 0.95)
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "pipeline": "tex2",
                "model": "xgboost",
                "data": "dm_only",
                "best_params": "{'max_depth': 3}",
            }
        )
    got = tuned_studies(client, "deepfake_source_bias", "gan_only")
    assert [s["pipeline"] for s in got] == ["col2"]
    assert got[0]["best_params"] == {"max_depth": 10}


def test_protocol_1_studies_are_ignored(client):
    """Исследования без первой попытки-умолчания не должны переоцениваться."""
    mlflow.set_experiment("deepfake_source_bias_tune")
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "pipeline": "col2",
                "model": "xgboost",
                "data": "gan_only",
                "best_params": "{'max_depth': 10}",
            }
        )  # протокол 1: тега нет
        mlflow.log_metric("best_val_auc", 0.95)
    assert tuned_studies(client, "deepfake_source_bias", "gan_only") == []
