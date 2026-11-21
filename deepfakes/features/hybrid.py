"""Hybrid feature extractor — конкатенация нескольких extractor-ов с нормализацией.

Используется ML-6 (color+texture), ROB-1 (FFT+LBP) и потенциально другими
комбинациями из таблицы.
"""

from __future__ import annotations

import cv2
import numpy as np
from tqdm.auto import tqdm

from deepfakes.features.base import _PARALLEL_MIN_IMAGES, BaseFeatureExtractor
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


class HybridExtractor(BaseFeatureExtractor):
    """Конкатенирует выходы нескольких BaseFeatureExtractor-ов.

    Каждый файл читается с диска ОДИН раз и раздаётся под-экстракторам через
    их `_prepare_raw` (раньше каждый под-экстрактор читал и декодировал файл
    заново — N чтений на изображение). Примечание: grayscale-подготовка идёт
    через cvtColor от цветного кадра, что может отличаться от прямого
    IMREAD_GRAYSCALE на ±1 уровень яркости из-за округления.

    Args:
        extractors: список уже сконструированных extractor-ов.
        names: опциональные короткие имена частей (для отладки).
    """

    def __init__(self, extractors: list[BaseFeatureExtractor], names: list[str] | None = None):
        # Все sub-extractors должны принимать пути; img_size совпадать не обязан.
        first = extractors[0]
        super().__init__(img_size=first.img_size, grayscale=first.grayscale, verbose=first.verbose)
        self.extractors = list(extractors)
        self.names = names or [type(e).__name__ for e in extractors]
        for e in self.extractors:
            e.verbose = False  # tqdm только снаружи

    def _extract(self, img: np.ndarray) -> np.ndarray:
        # Не используется: переопределяем transform() ниже, т.к. подвыделители
        # хотят пути файлов, а не уже считанный массив.
        raise NotImplementedError

    def _extract_one(self, path: str) -> np.ndarray | None:
        """Прочитать файл один раз, прогнать через все под-экстракторы."""
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            log.warning(f"  [extract failed] {path}: can't read image")
            return None
        try:
            parts = [e._extract(e._prepare_raw(raw)) for e in self.extractors]
            return np.concatenate(parts).astype(np.float32)
        except Exception as e:
            log.warning(f"  [extract failed] {path}: {e}")
            return None

    def transform(self, X) -> np.ndarray:
        n_jobs = getattr(self, "n_jobs", 1) or 1
        parallel = n_jobs not in (0, 1) and len(X) >= _PARALLEL_MIN_IMAGES
        if parallel:
            from joblib import Parallel, delayed

            gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(self._extract_one)(p) for p in X
            )
            feats = list(tqdm(gen, total=len(X), desc="HybridExtractor", disable=not self.verbose))
        else:
            iterator = tqdm(X, desc="HybridExtractor", disable=not self.verbose)
            feats = [self._extract_one(p) for p in iterator]
        X_out = self._finalize(feats)
        self._ensure_feature_names()
        return X_out

    def _build_feature_names(self) -> list[str] | None:
        """Конкатенация имён под-экстракторов с префиксом имени части ('{name}__{feat}').

        Под-экстракторы вызываются через _extract напрямую (их transform не
        исполняется), поэтому имена каждой части строим сами через её
        _build_feature_names; если хоть одна часть имён не даёт — None
        (тогда get_feature_names вернёт None, а не рассинхронизированный список)."""
        parts: list[str] = []
        for name, e in zip(self.names, self.extractors):
            sub = e.get_feature_names()
            if sub is None:
                try:
                    sub = e._build_feature_names()
                except Exception:
                    sub = None
            if sub is None:
                return None
            parts.extend(f"{name}__{s}" for s in sub)
        return parts
