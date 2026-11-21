"""Реестр пайплайнов: имя из конфига → класс.

Каждый класс несёт атрибут `group` (для group-runner и группировки в отчётах):
  - 'color'    — COL-1, COL-2, COL-3
  - 'freq'     — FREQ-2, FREQ-3, FREQ-4
  - 'grad'     — GRAD-1, GRAD-3, GRAD-4
  - 'texture'  — TEX-2, TEX-4, TEX-6
  - 'noise'    — NOISE-1, NOISE-2, NOISE-3
  - 'ml'       — ML-6, ML-7, ML-8 (hybrid feature engineering + специфические модели)
  - 'rob'      — ROB-1, ROB-2 (через scripts.run_score_fusion), ROB-9
  - 'dl'       — CNN/ViT
"""

from __future__ import annotations

from deepfakes.pipelines.col1 import Col1Pipeline
from deepfakes.pipelines.col2 import Col2Pipeline
from deepfakes.pipelines.col3 import Col3Pipeline
from deepfakes.pipelines.deep_lp import DeepLinearProbePipeline
from deepfakes.pipelines.ela import ElaPipeline
from deepfakes.pipelines.freq1 import Freq1Pipeline
from deepfakes.pipelines.freq2 import Freq2Pipeline
from deepfakes.pipelines.freq3 import Freq3Pipeline
from deepfakes.pipelines.freq4 import Freq4Pipeline
from deepfakes.pipelines.freq5 import Freq5Pipeline
from deepfakes.pipelines.grad1 import Grad1Pipeline
from deepfakes.pipelines.grad2 import Grad2Pipeline
from deepfakes.pipelines.grad3 import Grad3Pipeline
from deepfakes.pipelines.grad4 import Grad4Pipeline
from deepfakes.pipelines.ml6_hybrid import ML6HybridPipeline
from deepfakes.pipelines.ml7_freq_lbp import ML7FreqLBPPipeline
from deepfakes.pipelines.ml8_anomaly import ML8AnomalyPipeline
from deepfakes.pipelines.noise1 import Noise1Pipeline
from deepfakes.pipelines.noise2 import Noise2Pipeline
from deepfakes.pipelines.noise3 import Noise3Pipeline
from deepfakes.pipelines.rob1_fusion import Rob1FusionPipeline
from deepfakes.pipelines.tex1 import Tex1Pipeline
from deepfakes.pipelines.tex2 import Tex2Pipeline
from deepfakes.pipelines.tex3 import Tex3Pipeline
from deepfakes.pipelines.tex4 import Tex4Pipeline
from deepfakes.pipelines.tex5 import Tex5Pipeline
from deepfakes.pipelines.tex6 import Tex6Pipeline

# Проставляем group у тех классов, у которых она не задана в файле
Col2Pipeline.group = "color"  # type: ignore[attr-defined]
Freq4Pipeline.group = "freq"  # type: ignore[attr-defined]
Grad4Pipeline.group = "grad"  # type: ignore[attr-defined]
Tex2Pipeline.group = "texture"  # type: ignore[attr-defined]
Noise1Pipeline.group = "noise"  # type: ignore[attr-defined]
ML6HybridPipeline.group = "ml"  # type: ignore[attr-defined]


REGISTRY: dict[str, type] = {
    "col1": Col1Pipeline,
    "deep_lp": DeepLinearProbePipeline,
    "col2": Col2Pipeline,
    "col3": Col3Pipeline,
    "freq1": Freq1Pipeline,
    "freq2": Freq2Pipeline,
    "freq3": Freq3Pipeline,
    "freq4": Freq4Pipeline,
    "freq5": Freq5Pipeline,
    "grad1": Grad1Pipeline,
    "grad2": Grad2Pipeline,
    "grad3": Grad3Pipeline,
    "grad4": Grad4Pipeline,
    "tex1": Tex1Pipeline,
    "tex2": Tex2Pipeline,
    "tex3": Tex3Pipeline,
    "tex4": Tex4Pipeline,
    "tex5": Tex5Pipeline,
    "tex6": Tex6Pipeline,
    "noise1": Noise1Pipeline,
    "noise2": Noise2Pipeline,
    "noise3": Noise3Pipeline,
    "ela": ElaPipeline,
    "ml6_hybrid": ML6HybridPipeline,
    "ml7_freq_lbp": ML7FreqLBPPipeline,
    "ml8_anomaly": ML8AnomalyPipeline,
    "rob1_fusion": Rob1FusionPipeline,
}

# Опциональные DL-пайплайны (лениво, чтобы не требовать torch).
try:
    from deepfakes.pipelines.dl import DLPipeline

    REGISTRY["dl_efficientnet"] = DLPipeline
    REGISTRY["dl_vit_tiny"] = DLPipeline
    DLPipeline.group = "dl"  # type: ignore[attr-defined]
except ImportError as e:
    # Не молчим: иначе опечатка в собственном коде dl.py «выключает» DL-пайплайны
    # без единого сообщения.
    import logging

    logging.getLogger(__name__).info(f"DL pipelines unavailable (no torch/timm?): {e}")


def get_pipeline_class(name: str) -> type:
    if name not in REGISTRY:
        raise KeyError(f"Unknown pipeline '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def pipelines_by_group(group: str) -> list[str]:
    """Все имена пайплайнов в данной группе."""
    return sorted(name for name, cls in REGISTRY.items() if getattr(cls, "group", None) == group)


def all_groups() -> list[str]:
    return sorted({getattr(cls, "group", "?") for cls in REGISTRY.values()})
