"""FREQ-3: multi-level wavelet → статистики по subbands."""

from __future__ import annotations

from deepfakes.features.freq import WaveletExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Freq3Pipeline(BasePipeline):
    pipeline_id = "FREQ-3"
    group = "freq"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = WaveletExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            wavelet=pipeline_cfg.get("wavelet", "db2"),
            level=pipeline_cfg.get("level", 3),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
