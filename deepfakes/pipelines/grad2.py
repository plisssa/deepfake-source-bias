"""GRAD-2: Scharr magnitude + orientation histograms (зеркальный аналог GRAD-1)."""

from __future__ import annotations

from deepfakes.features.grad import ScharrExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Grad2Pipeline(BasePipeline):
    pipeline_id = "GRAD-2"
    group = "grad"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = ScharrExtractor(
            img_size=pipeline_cfg.get("img_size", 128),
            patch_grid=pipeline_cfg.get("patch_grid", 4),
            n_orient_bins=pipeline_cfg.get("n_orient_bins", 9),
            threshold=pipeline_cfg.get("threshold", None),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
