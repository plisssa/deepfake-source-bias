"""FREQ-2: блочный DCT (8×8 / 16×16) → агрегаты по блокам."""

from __future__ import annotations

from deepfakes.features.freq import DCTBlockExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Freq2Pipeline(BasePipeline):
    pipeline_id = "FREQ-2"
    group = "freq"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = DCTBlockExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            block_size=pipeline_cfg.get("block_size", 8),
            log_scale=pipeline_cfg.get("log_scale", True),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
