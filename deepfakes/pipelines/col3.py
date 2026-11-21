"""COL-3: YCbCr Cb/Cr moments + chi-square distance local vs global."""

from __future__ import annotations

from deepfakes.features.color import YCbCrChiSquareExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Col3Pipeline(BasePipeline):
    pipeline_id = "COL-3"
    group = "color"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = YCbCrChiSquareExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            n_bins=pipeline_cfg.get("n_bins", 32),
            patch_grid=pipeline_cfg.get("patch_grid", 4),
            full_stats=pipeline_cfg.get("full_stats", True),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
