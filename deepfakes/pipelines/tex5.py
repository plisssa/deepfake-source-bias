"""TEX-5: банк фильтров Габора."""

from __future__ import annotations

from deepfakes.features.texture import GaborBankExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class Tex5Pipeline(BasePipeline):
    pipeline_id = "TEX-5"
    group = "texture"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = GaborBankExtractor(
            img_size=pipeline_cfg.get("img_size", 128),
            orientations=tuple(pipeline_cfg.get("orientations", [0, 45, 90, 135])),
            wavelengths=tuple(pipeline_cfg.get("wavelengths", [4, 8, 16])),
            gamma=pipeline_cfg.get("gamma", 0.5),
            psi=pipeline_cfg.get("psi", 0.0),
            patch_grid=pipeline_cfg.get("patch_grid", 1),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
