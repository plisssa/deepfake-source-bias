"""TEX-3: Local Ternary Patterns (LTP)."""

from __future__ import annotations

from deepfakes.features.texture import LTPExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Tex3Pipeline(BasePipeline):
    pipeline_id = "TEX-3"
    group = "texture"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = LTPExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            radii=tuple(pipeline_cfg.get("radii", [1, 2])),
            t=pipeline_cfg.get("t", 5.0),
            n_bins=pipeline_cfg.get("n_bins", 256),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
