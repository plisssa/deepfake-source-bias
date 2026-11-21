"""Gradient descriptors for classical image classifiers."""

from __future__ import annotations

from deepfakes.features.grad import LoGExtractor, LoGFlattenExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Grad4Pipeline(BasePipeline):
    pipeline_id = "GRAD-4"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        mode = pipeline_cfg.get("mode", "patch")
        if mode == "patch":
            extractor = LoGExtractor(
                img_size=pipeline_cfg.get("img_size", 128),
                sigma=pipeline_cfg.get("sigma", 1.2),
                zc_threshold=pipeline_cfg.get("zc_threshold", 0.5),
                patch_size=pipeline_cfg.get("patch_size", 16),
                full_stats=pipeline_cfg.get("full_stats", True),
            )
        elif mode == "flatten":
            extractor = LoGFlattenExtractor(
                img_size=pipeline_cfg.get("img_size", 128),
                sigma=pipeline_cfg.get("sigma", 1.2),
            )
        else:
            raise ValueError(f"GRAD-4 mode must be 'patch' or 'flatten', got {mode}")
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
