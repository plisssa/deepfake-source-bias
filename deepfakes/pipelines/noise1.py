"""NOISE-1 pipeline: noise residual + SRM features + классификатор.

В таблице рекомендована SVM, но реализация поддерживает любую модель из registry,
чтобы войти в единый бенчмарк (LR / RF / XGBoost).
"""

from __future__ import annotations

from deepfakes.features.noise import NoiseResidualSRMExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Noise1Pipeline(BasePipeline):
    pipeline_id = "NOISE-1"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = NoiseResidualSRMExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            denoise_method=pipeline_cfg.get("denoise_method", "nlmeans"),
            denoise_strength=pipeline_cfg.get("denoise_strength", 10.0),
            use_srm=pipeline_cfg.get("use_srm", True),
            use_residual=pipeline_cfg.get("use_residual", True),
            patch_grid=pipeline_cfg.get("patch_grid", 1),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
