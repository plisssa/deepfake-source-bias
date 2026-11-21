"""ROB-1: hybrid features + scaling + PCA.

Конфиг описывает список компонент. Например:
    components:
      - kind: freq        # один из ['color', 'freq', 'grad_log', 'grad_sobel', 'lbp', 'hog', 'noise']
        cfg: {n_bins: 8}
      - kind: lbp
        cfg: {radii: [1, 2], method: uniform}
    fusion:
      scale: true
      variance_threshold: 0.0
      pca_components: 64
"""

from __future__ import annotations

from deepfakes.features.color import ColorMomentsExtractor
from deepfakes.features.deep import DeepFeatureExtractor
from deepfakes.features.freq import FullFFTExtractor, HFEnergyExtractor
from deepfakes.features.fusion import FeatureFusionExtractor
from deepfakes.features.grad import LoGExtractor, ScharrExtractor, SobelExtractor
from deepfakes.features.noise import NoiseResidualSRMExtractor
from deepfakes.features.texture import (
    GaborBankExtractor,
    HOGExtractor,
    LBPExtractor,
    LPQExtractor,
    LTPExtractor,
)
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.base import BasePipeline

_FACTORY = {
    "color": lambda cfg: ColorMomentsExtractor(**cfg),
    "freq": lambda cfg: HFEnergyExtractor(**cfg),
    "fft": lambda cfg: FullFFTExtractor(**cfg),
    "grad_log": lambda cfg: LoGExtractor(**cfg),
    "grad_sobel": lambda cfg: SobelExtractor(**cfg),
    "grad_scharr": lambda cfg: ScharrExtractor(**cfg),
    "lbp": lambda cfg: LBPExtractor(**cfg),
    "ltp": lambda cfg: LTPExtractor(**cfg),
    "lpq": lambda cfg: LPQExtractor(**cfg),
    "gabor": lambda cfg: GaborBankExtractor(**cfg),
    "hog": lambda cfg: HOGExtractor(**cfg),
    "noise": lambda cfg: NoiseResidualSRMExtractor(**cfg),
    # deep-эмбеддинги замороженного бэкбона (нужны extras [dl]: torch+timm) —
    "deep": lambda cfg: DeepFeatureExtractor(**cfg),
}


class Rob1FusionPipeline(BasePipeline):
    pipeline_id = "ROB-1"
    group = "rob"

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        components = pipeline_cfg.get("components", [])
        if not components:
            raise ValueError("ROB-1 requires non-empty 'components' list")
        subs = []
        names: list[str] = []
        for comp in components:
            kind = comp["kind"]
            cfg = dict(comp.get("cfg", {}))
            if kind not in _FACTORY:
                raise ValueError(
                    f"Unknown fusion component kind: {kind}. Known: {sorted(_FACTORY)}"
                )
            subs.append(_FACTORY[kind](cfg))
            names.append(comp.get("name", kind))
        fusion_cfg = pipeline_cfg.get("fusion", {})
        extractor = FeatureFusionExtractor(
            sub_extractors=subs,
            names=names,
            scale=fusion_cfg.get("scale", True),
            variance_threshold=fusion_cfg.get("variance_threshold", None),
            pca_components=fusion_cfg.get("pca_components", None),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            augment_cfg=pipeline_cfg.get("augment"),
        )
