"""GRAD-3: Canny edges → density + orientation hist + global statistics."""

from __future__ import annotations

from deepfakes.features.grad import CannyEdgeExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Grad3Pipeline(BasePipeline):
    pipeline_id = "GRAD-3"
    group = "grad"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = CannyEdgeExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            low_threshold=pipeline_cfg.get("low_threshold", 50),
            high_threshold=pipeline_cfg.get("high_threshold", 150),
            aperture_size=pipeline_cfg.get("aperture_size", 3),
            patch_grid=pipeline_cfg.get("patch_grid", 4),
            n_orient_bins=pipeline_cfg.get("n_orient_bins", 9),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
