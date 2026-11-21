"""GRAD-1: Sobel magnitude + orientation histograms (компактный counterpart к GRAD-4)."""

from __future__ import annotations

from deepfakes.features.grad import SobelExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Grad1Pipeline(BasePipeline):
    pipeline_id = "GRAD-1"
    group = "grad"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = SobelExtractor(
            img_size=pipeline_cfg.get("img_size", 128),
            ksize=pipeline_cfg.get("ksize", 3),
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
