"""FREQ-5: 1D CWT (Morlet / Mexican Hat) над проекциями изображения."""

from __future__ import annotations

from deepfakes.features.freq import CWTScalogramExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Freq5Pipeline(BasePipeline):
    pipeline_id = "FREQ-5"
    group = "freq"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = CWTScalogramExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            wavelet=pipeline_cfg.get("wavelet", "morl"),
            scales=tuple(pipeline_cfg.get("scales", (2, 4, 8, 16))),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
