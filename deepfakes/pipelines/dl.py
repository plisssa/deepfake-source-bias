"""DL-pipeline: CNN/ViT baseline через timm. Принимает пути напрямую, без extractor-а."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from deepfakes.eval.metrics import (
    ClassificationReport,
    compute_report,
    plot_calibration,
    plot_confusion_matrix,
    plot_pr,
    plot_roc,
    tune_threshold,
)
from deepfakes.models.cnn_baseline import DLBackbone, DLConfig
from deepfakes.pipelines.base import BasePipeline, PipelineArtifacts
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


class DLPipeline:
    """Тонкая обёртка, повторяющая интерфейс BasePipeline, но для CNN/ViT."""

    pipeline_id = "DL"

    def __init__(self, cfg: DLConfig):
        self.cfg = cfg
        self.backbone: DLBackbone | None = None
        self.threshold: float = 0.5

    @classmethod
    def from_cfg(cls, pipeline_cfg: Any, model_spec=None, cache_dir=None):
        known = {f for f in DLConfig.__dataclass_fields__}
        cfg = DLConfig(**{k: v for k, v in pipeline_cfg.items() if k in known})
        return cls(cfg=cfg)

    def fit(
        self,
        X_paths,
        y,
        X_val_paths=None,
        y_val=None,
        threshold_objective: str = "f1",
    ) -> PipelineArtifacts:
        self.backbone = DLBackbone(self.cfg)
        self.backbone.fit(X_paths, y)
        train_score = self.backbone.predict_proba(X_paths)
        val_report = None
        if X_val_paths is not None and y_val is not None:
            scores = self.backbone.predict_proba(X_val_paths)
            self.threshold = tune_threshold(y_val, scores, objective=threshold_objective)
            val_report = compute_report(y_val, scores, threshold=self.threshold)
        train_report = compute_report(y, train_score, threshold=self.threshold)
        # Контракт BasePipeline: fit возвращает PipelineArtifacts (раньше — None,
        # и run_pipeline падал на artifacts.feature_dim)
        return PipelineArtifacts(
            extractor=None,
            model=self.backbone,
            feature_dim=0,
            threshold=self.threshold,
            train_report=train_report,
            val_report=val_report,
        )

    def predict_proba(self, X_paths, return_valid_mask: bool = False):
        assert self.backbone is not None
        scores = self.backbone.predict_proba(X_paths)
        if return_valid_mask:
            return scores, np.ones(len(scores), dtype=bool)
        return scores

    # Gap-мониторинг — та же логика, что у классических пайплайнов
    compute_gaps = staticmethod(BasePipeline.compute_gaps)
    log_gaps_to_mlflow = BasePipeline.log_gaps_to_mlflow

    def save(self, out_path) -> Path:
        """Сохранить веса backbone (контракт BasePipeline.save)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assert self.backbone is not None
        self.backbone.save(out_path)
        return out_path

    def evaluate(self, X_paths, y, threshold: float | None = None) -> ClassificationReport:
        t = self.threshold if threshold is None else threshold
        scores = self.predict_proba(X_paths)
        return compute_report(y, scores, threshold=t)

    def log_split_to_mlflow(self, name, y_true, y_score, out_dir, threshold=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t = self.threshold if threshold is None else threshold
        report = compute_report(y_true, y_score, threshold=t)
        mlflow.log_metrics(report.flatten_for_mlflow(prefix=f"{name}_"))
        for fn, suffix in [
            (
                lambda: plot_confusion_matrix(report.confusion_matrix, out_dir / f"cm_{name}.png"),
                "cm",
            ),
            (lambda: plot_roc(y_true, y_score, out_dir / f"roc_{name}.png"), "roc"),
            (lambda: plot_pr(y_true, y_score, out_dir / f"pr_{name}.png"), "pr"),
            (lambda: plot_calibration(y_true, y_score, out_dir / f"cal_{name}.png"), "cal"),
        ]:
            p = fn()
            mlflow.log_artifact(str(p))
        json_path = out_dir / f"report_{name}.json"
        report.to_json(json_path)
        mlflow.log_artifact(str(json_path))
        return report
