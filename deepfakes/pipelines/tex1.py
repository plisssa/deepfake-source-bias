"""TEX-1: одиночный канонический LBP (uniform, один радиус).

Baseline текстурного блока: тот же LBPExtractor, что и в TEX-2, но без
multi-scale — один радиус (default radii=[1]).
"""

from __future__ import annotations

from deepfakes.features.texture import LBPExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Tex1Pipeline(BasePipeline):
    pipeline_id = "TEX-1"
    group = "texture"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = LBPExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            radii=tuple(pipeline_cfg.get("radii", [1])),
            n_points=pipeline_cfg.get("n_points", None),
            method=pipeline_cfg.get("method", "uniform"),
            block_grid=pipeline_cfg.get("block_grid", None),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
