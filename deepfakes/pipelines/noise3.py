"""NOISE-3: PRNU-подобный single-image residual."""

from __future__ import annotations

from deepfakes.features.noise import PRNUExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Noise3Pipeline(BasePipeline):
    pipeline_id = "NOISE-3"
    group = "noise"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = PRNUExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            denoise_method=pipeline_cfg.get("denoise_method", "nlmeans"),
            denoise_strength=pipeline_cfg.get("denoise_strength", 8.0),
            patch_grid=pipeline_cfg.get("patch_grid", 2),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
