"""Frozen vision backbones with trainable classifier heads."""

from __future__ import annotations

from deepfakes.features.deep import DeepFeatureExtractor
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline


class DeepLinearProbePipeline(BasePipeline):
    pipeline_id = "DEEP-LP"
    group = "learned"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = DeepFeatureExtractor(
            backbone=pipeline_cfg["backbone"],
            img_size=int(pipeline_cfg.get("img_size", 224)),
            batch_size=int(pipeline_cfg.get("batch_size", 64)),
            resize_mode=pipeline_cfg.get("resize_mode", "center_crop"),
            normalize=pipeline_cfg.get("normalize", "model"),
            model_img_size=pipeline_cfg.get("model_img_size"),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
