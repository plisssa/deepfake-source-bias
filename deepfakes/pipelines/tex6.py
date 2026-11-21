"""TEX-6: GLCM / Haralick — статистическая текстура."""

from __future__ import annotations

import numpy as np

from deepfakes.features.texture import GLCMExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Tex6Pipeline(BasePipeline):
    pipeline_id = "TEX-6"
    group = "texture"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        angles = pipeline_cfg.get("angles", None)
        if angles is None:
            angles = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
        extractor = GLCMExtractor(
            img_size=pipeline_cfg.get("img_size", 128),
            distances=tuple(pipeline_cfg.get("distances", [1, 2, 3])),
            angles=tuple(angles),
            levels=pipeline_cfg.get("levels", 32),
            patch_grid=pipeline_cfg.get("patch_grid", 1),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
