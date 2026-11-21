"""NOISE-2: JPEG blockiness."""

from __future__ import annotations

from deepfakes.features.noise import JPEGBlockinessExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Noise2Pipeline(BasePipeline):
    pipeline_id = "NOISE-2"
    group = "noise"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = JPEGBlockinessExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            block_size=pipeline_cfg.get("block_size", 8),
            patch_grid=pipeline_cfg.get("patch_grid", 4),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
