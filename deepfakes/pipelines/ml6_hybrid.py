"""ML-6: hybrid color+texture features → CatBoost.

В таблице ML-6 = CatBoost на гибриде color+texture. Этот пайплайн собирает
ColorMomentsExtractor + LBPExtractor и подаёт в CatBoost (по умолчанию;
любая модель из registry тоже работает).
"""

from __future__ import annotations

from deepfakes.features.color import ColorMomentsExtractor
from deepfakes.features.hybrid import HybridExtractor
from deepfakes.features.texture import LBPExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class ML6HybridPipeline(BasePipeline):
    pipeline_id = "ML-6"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        color = ColorMomentsExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            patch_sizes=tuple(pipeline_cfg.get("color_patch_sizes", [16, 32])),
            colorspace=pipeline_cfg.get("colorspace", "YCbCr"),
            full_stats=pipeline_cfg.get("color_full_stats", True),
        )
        texture = LBPExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            radii=tuple(pipeline_cfg.get("lbp_radii", [1, 2])),
            method=pipeline_cfg.get("lbp_method", "uniform"),
        )
        extractor = HybridExtractor([color, texture], names=["color", "texture"])
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
