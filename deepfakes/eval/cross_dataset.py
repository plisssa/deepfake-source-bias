"""Cross-domain transfer evaluation on two source datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np

from deepfakes.eval.metrics import ClassificationReport, compute_report
from deepfakes.io.collect import DatasetBundle
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import sanitize_mlflow_name
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CrossDatasetResult:
    train_bundle_name: str
    test_bundle_name: str
    report: ClassificationReport


def run_cross_dataset(
    train_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    pipeline_name: str,
    pipeline_cfg: dict,
    model_spec: ModelSpec,
    cache_dir: str | None = None,
    out_dir: str | Path | None = None,
    seed: int = 42,
) -> CrossDatasetResult:
    log.info(
        f"[cross-dataset] train={'+'.join(s.name for s in train_bundle.real_sources + train_bundle.fake_sources)} "
        f"→ test={'+'.join(s.name for s in test_bundle.real_sources + test_bundle.fake_sources)}"
    )
    X_train, y_train, _, _ = train_bundle.collect(seed=seed)
    X_test, y_test, _, _ = test_bundle.collect(seed=seed)

    # Дубликаты между train- и test-бандлами — тихая утечка cross-протокола
    overlap = set(X_train.tolist()) & set(X_test.tolist())
    if overlap:
        raise ValueError(
            f"[cross-dataset] {len(overlap)} файлов встречаются и в train-, и в "
            f"test-бандле (пример: {next(iter(overlap))}) — протокол некорректен"
        )

    pipeline_cls = get_pipeline_class(pipeline_name)
    pipe: BasePipeline = pipeline_cls.from_cfg(pipeline_cfg, model_spec, cache_dir=cache_dir)
    pipe.fit(X_train, y_train)
    scores, valid = pipe.predict_proba(X_test, return_valid_mask=True)
    y_test = np.asarray(y_test)
    if not valid.all():
        log.warning(f"[cross-dataset] dropping {int((~valid).sum())} unreadable test images")
        scores, y_test = scores[valid], y_test[valid]
    report = compute_report(y_test, scores, threshold=pipe.threshold)

    train_name = "+".join(s.name for s in train_bundle.real_sources + train_bundle.fake_sources)
    test_name = "+".join(s.name for s in test_bundle.real_sources + test_bundle.fake_sources)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # '+' и другие символы недопустимы в именах MLflow-метрик — санитизируем
        tag = sanitize_mlflow_name(f"{train_name}__to__{test_name}")
        rep = pipe.log_split_to_mlflow(f"cross_{tag}", y_test, scores, out_dir)
        if mlflow.active_run() is not None:
            mlflow.log_metric(f"cross_{tag}_auc", rep.roc_auc)
        return CrossDatasetResult(train_name, test_name, rep)

    return CrossDatasetResult(train_name, test_name, report)
