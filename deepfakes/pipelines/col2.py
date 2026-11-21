"""COL-2 pipeline: multi-scale color moments (с kurtosis/entropy) + один из канонических классификаторов."""

from __future__ import annotations

from deepfakes.features.color import ColorMomentsExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Col2Pipeline(BasePipeline):
    pipeline_id = "COL-2"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = ColorMomentsExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            patch_sizes=tuple(pipeline_cfg.get("patch_sizes", [32])),
            colorspace=pipeline_cfg.get("colorspace", "RGB"),
            full_stats=pipeline_cfg.get("full_stats", True),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
