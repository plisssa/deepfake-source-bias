"""FOR-1: Error Level Analysis (ELA)."""

from __future__ import annotations

from deepfakes.features.forensic import ELAExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class ElaPipeline(BasePipeline):
    pipeline_id = "FOR-1"
    group = "noise"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = ELAExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            quality=pipeline_cfg.get("quality", 90),
            patch_grid=pipeline_cfg.get("patch_grid", 4),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
