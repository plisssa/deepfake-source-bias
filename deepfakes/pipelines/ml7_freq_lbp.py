"""ML-7: гибрид FFT-частотных + LBP-текстурных признаков → MLP."""

from __future__ import annotations

from deepfakes.features.freq import HFEnergyExtractor
from deepfakes.features.hybrid import HybridExtractor
from deepfakes.features.texture import LBPExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class ML7FreqLBPPipeline(BasePipeline):
    pipeline_id = "ML-7"
    group = "ml"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        freq = HFEnergyExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            n_bins=pipeline_cfg.get("freq_n_bins", 8),
            include_slope=True,
            include_flatness=True,
            include_basic_stats=True,
        )
        lbp = LBPExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            radii=tuple(pipeline_cfg.get("lbp_radii", [1, 2])),
            method=pipeline_cfg.get("lbp_method", "uniform"),
        )
        extractor = HybridExtractor([freq, lbp], names=["freq", "lbp"])
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
