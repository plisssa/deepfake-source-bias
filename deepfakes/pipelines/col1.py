"""COL-1: канальные гистограммы (RGB/HSV/YCbCr/Lab) — простой baseline."""

from __future__ import annotations

from deepfakes.features.color import ColorHistogramExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Col1Pipeline(BasePipeline):
    pipeline_id = "COL-1"
    group = "color"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = ColorHistogramExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            n_bins=pipeline_cfg.get("n_bins", 32),
            colorspace=pipeline_cfg.get("colorspace", "RGB"),
            patch_grid=pipeline_cfg.get("patch_grid", 1),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
