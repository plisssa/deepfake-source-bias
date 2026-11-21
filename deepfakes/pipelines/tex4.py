"""TEX-4: HOG (Histogram of Oriented Gradients)."""

from __future__ import annotations

from deepfakes.features.texture import HOGExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Tex4Pipeline(BasePipeline):
    pipeline_id = "TEX-4"
    group = "texture"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = HOGExtractor(
            img_size=pipeline_cfg.get("img_size", 128),
            orientations=pipeline_cfg.get("orientations", 9),
            pixels_per_cell=pipeline_cfg.get("pixels_per_cell", 8),
            cells_per_block=pipeline_cfg.get("cells_per_block", 2),
            block_norm=pipeline_cfg.get("block_norm", "L2-Hys"),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
