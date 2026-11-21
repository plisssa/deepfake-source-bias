"""Единый MLflow setup: tracking URI, эксперимент, теги, артефакты."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import yaml

from deepfakes.utils.git import git_info
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

# MLflow допускает в именах метрик/параметров только: буквы, цифры, _ - . пробел /
_INVALID_MLFLOW_CHARS = re.compile(r"[^A-Za-z0-9_\-\. /]")


def sanitize_mlflow_name(name: str) -> str:
    """Заменить недопустимые для MLflow символы ('+', ':', и т.п.) на '-'."""
    return _INVALID_MLFLOW_CHARS.sub("-", name)


def init_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "default",
) -> str:
    """Инициализировать MLflow tracking.

    Если tracking_uri не задан, читаем из env MLFLOW_TRACKING_URI;
    если и там пусто — fallback на локальный ./mlruns.
    """
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    # MLflow >= 3.x запрещает файловый бэкенд без явного opt-out; не роняем
    # запуски пользователей со старым file:-URI
    if tracking_uri.startswith(("file:", "./")):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    log.info(f"MLflow URI: {tracking_uri} | experiment: {experiment_name}")
    return tracking_uri


def default_tags(extra: dict[str, Any] | None = None) -> dict[str, str]:
    """Собрать стандартные теги: git, pipeline, dataset, seed."""
    tags = git_info()
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


def log_dict_as_yaml(d: dict, artifact_name: str = "config.yaml") -> None:
    """Залогировать словарь как YAML-артефакт.

    Пишем во временную директорию (уникальную на вызов): прежний вариант с
    общей ./.tmp_artifact_ создавал мусорные локальные папки при remote-URI и
    гонку при параллельных ранах."""
    with tempfile.TemporaryDirectory(prefix="mlflow_yaml_") as tmpdir:
        tmp = Path(tmpdir) / artifact_name
        tmp.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
        mlflow.log_artifact(str(tmp))


@contextmanager
def mlflow_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Контекст для одного MLflow run-а с автоматическими тегами."""
    base_tags = default_tags()
    if tags:
        base_tags.update({k: str(v) for k, v in tags.items()})
    with mlflow.start_run(run_name=run_name, tags=base_tags, nested=nested) as run:
        yield run


def pipeline_variant(cfg: Any) -> str:
    """Имя ВАРИАНТА пайплайна для тегов и имён прогонов — имя файла конфига.

    `cfg.pipeline.name` — это имя КЛАССА (по нему реестр находит реализацию), и
    разные конфиги могут делить один класс. Так устроен `rob_deep_fusion.yaml`:
    ViT-эмбеддинги + классические признаки поверх класса ROB-1, поэтому
    `name: rob1_fusion`. Пока тегом служило имя класса, deep-гибрид логировался
    как `pipeline=rob1_fusion`, и дедупликация отчёта (`keep="last"`) 28.09.2026
    подменила им классический rob1_fusion: в таблице GAN-домена вместо 0.8322
    оказалось 0.8979 от совсем другого пайплайна.

    Имя выбранного конфига берётся из Hydra (`runtime.choices["pipeline"]`); вне
    Hydra — откат на имя класса."""
    try:
        from hydra.core.hydra_config import HydraConfig

        choice = HydraConfig.get().runtime.choices.get("pipeline")
        if choice:
            return str(choice)
    except Exception:  # вне hydra.main (тесты, ноутбуки)
        pass
    return str(cfg.pipeline.name)
