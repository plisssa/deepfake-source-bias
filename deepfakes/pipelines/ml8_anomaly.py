"""ML-8: One-Class SVM на NOISE-3 (PRNU)-фичах. Anomaly detection.

Особенность: вместо обычной модели использует OneClassPredictor — обучается
ТОЛЬКО на real-сэмплах (y==0). model_spec игнорируется (для единообразия
интерфейса принимается, но всегда строится OneClassPredictor).
"""

from __future__ import annotations

from deepfakes.features.noise import NoiseResidualSRMExtractor, PRNUExtractor
from deepfakes.models.anomaly import OneClassPredictor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class ML8AnomalyPipeline(BasePipeline):
    pipeline_id = "ML-8"
    group = "ml"

    @classmethod
    def from_cfg(
        cls, pipeline_cfg, model_spec: ModelSpec | None = None, cache_dir: str | None = None
    ):
        feat = pipeline_cfg.get("features", "prnu")
        if feat == "prnu":
            extractor = PRNUExtractor(
                img_size=pipeline_cfg.get("img_size", 256),
                denoise_method=pipeline_cfg.get("denoise_method", "nlmeans"),
                denoise_strength=pipeline_cfg.get("denoise_strength", 8.0),
                patch_grid=pipeline_cfg.get("patch_grid", 2),
            )
        elif feat == "srm":
            extractor = NoiseResidualSRMExtractor(
                img_size=pipeline_cfg.get("img_size", 256),
                patch_grid=pipeline_cfg.get("patch_grid", 2),
            )
        else:
            raise ValueError(f"ML-8 features must be 'prnu' or 'srm', got {feat}")
        # Кастомный spec: name='one_class_svm'
        spec = ModelSpec(name="one_class_svm", params=dict(pipeline_cfg.get("oc_params", {})))
        pipe = cls(extractor=extractor, model_spec=spec, cache_dir=cache_dir)
        return pipe

    def _fit_model(self, X, y) -> None:
        """Вместо фабрики classical.build_model — OneClassPredictor (real-only).

        Вся остальная логика (экстракция, кэш, drop невалидных, порог на val,
        отчёты) — общая, из BasePipeline.fit."""
        params = dict(self.model_spec.params)
        self.model = OneClassPredictor(
            nu=params.get("nu", 0.1),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            with_scaler=params.get("with_scaler", True),
        )
        self.model.fit(X, y)
