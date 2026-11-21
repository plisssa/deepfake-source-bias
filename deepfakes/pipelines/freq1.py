"""FREQ-1: FFT всего изображения → спектральный дескриптор (профили + slope + фаза)."""

from __future__ import annotations

from deepfakes.features.freq import FullFFTExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Freq1Pipeline(BasePipeline):
    pipeline_id = "FREQ-1"
    group = "freq"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = FullFFTExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            radial_bins=pipeline_cfg.get("radial_bins", 16),
            angular_bins=pipeline_cfg.get("angular_bins", 8),
            window=pipeline_cfg.get("window", "hann"),
            include_phase=pipeline_cfg.get("include_phase", True),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
