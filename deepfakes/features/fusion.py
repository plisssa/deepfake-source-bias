"""ROB-1: Feature fusion = HybridExtractor + StandardScaler + (опц.) PCA + VarianceThreshold."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from deepfakes.features.base import BaseFeatureExtractor
from deepfakes.features.hybrid import HybridExtractor
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


class FeatureFusionExtractor(BaseFeatureExtractor):
    """ROB-1: hybrid fusion с нормализацией и уменьшением размерности.

    pipeline на этапе fit:
        1. Конкатенация выходов нескольких extractor-ов (HybridExtractor).
        2. VarianceThreshold (опционально, по умолчанию выкл).
        3. StandardScaler.
        4. PCA (опционально, число компонент или None).

    Состояние scaler/pca фиксируется ТОЛЬКО в fit (BasePipeline вызывает его на
    train до любых transform — см. requires_fit). transform на не-обученном
    экстракторе — ошибка, а не тихий fit на чужих данных: раньше при попадании
    train-кэша scaler/PCA молча обучались на val/test (утечка + рассинхрон
    признакового пространства с кэшем).
    """

    #: выход зависит от fitted-состояния (scaler/PCA на конкретном train) —
    #: кэшировать по одним лишь путям и параметрам нельзя
    cacheable = False
    #: BasePipeline.fit обязан вызвать fit(train) до transform
    requires_fit = True

    def __init__(
        self,
        sub_extractors: list[BaseFeatureExtractor],
        names: list[str] | None = None,
        scale: bool = True,
        variance_threshold: float | None = None,
        pca_components: float | None = None,
        verbose: bool = True,
    ):
        first = sub_extractors[0]
        super().__init__(img_size=first.img_size, grayscale=first.grayscale, verbose=verbose)
        # контракт sklearn get_params/clone: параметры конструктора — атрибуты
        self.sub_extractors = sub_extractors
        self.names = names
        self.hybrid_ = HybridExtractor(sub_extractors, names=names)
        self.scale = bool(scale)
        self.variance_threshold = variance_threshold
        self.pca_components = pca_components
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.vt_: VarianceThreshold | None = None
        self._fitted = False
        self._train_paths_: np.ndarray | None = None
        self._cached_train_: np.ndarray | None = None
        self._train_valid_mask_: np.ndarray | None = None

    def _extract(self, img: np.ndarray) -> np.ndarray:  # not used
        raise NotImplementedError

    def fit(self, X, y=None):
        # Повторный fit (новый фолд CV) полностью сбрасывает прежнее состояние.
        self.scaler_ = None
        self.pca_ = None
        self.vt_ = None
        self._fitted = False
        self.feature_names_ = None

        F = self.hybrid_.transform(X)
        # VT/scaler/PCA фитим только на валидных строках: нулевые векторы
        # упавших изображений не должны портить статистики препроцессинга
        mask = getattr(self.hybrid_, "last_valid_mask_", None)
        if mask is None:
            mask = np.ones(F.shape[0], dtype=bool)
        if self.variance_threshold is not None:
            self.vt_ = VarianceThreshold(threshold=self.variance_threshold)
            self.vt_.fit(F[mask])
            F = self.vt_.transform(F)
        if self.scale:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(F[mask])
            F = self.scaler_.transform(F)
        if self.pca_components is not None:
            self.pca_ = PCA(n_components=self.pca_components, random_state=42)
            self.pca_.fit(F[mask])
            F = self.pca_.transform(F)
            log.info(
                f"  ROB-1 PCA: kept {F.shape[1]} components (was {self.hybrid_.n_features_out_})"
            )
        self._fitted = True
        self.n_features_out_ = F.shape[1]
        self._cached_train_ = F.astype(np.float32)
        self._train_paths_ = np.asarray(X).copy()
        # маска валидности train-выборки из hybrid (для BasePipeline)
        self.last_valid_mask_ = mask
        self._train_valid_mask_ = mask
        self._ensure_feature_names()
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "FeatureFusionExtractor.transform вызван до fit. "
                "Scaler/PCA должны обучаться только на train — вызовите fit(train) явно."
            )
        # Тот же train, что и в fit — отдаём уже посчитанные признаки,
        # не извлекая всё вторично (fit_transform-паттерн). Маску валидности
        # восстанавливаем из fit — иначе тут осталась бы маска от прошлого
        # transform (например, val), и BasePipeline дропнул бы не те строки.
        if (
            self._train_paths_ is not None
            and len(X) == len(self._train_paths_)
            and np.array_equal(np.asarray(X), self._train_paths_)
        ):
            self.last_valid_mask_ = self._train_valid_mask_
            return self._cached_train_
        F = self.hybrid_.transform(X)
        self.last_valid_mask_ = getattr(self.hybrid_, "last_valid_mask_", None)
        if self.vt_ is not None:
            F = self.vt_.transform(F)
        if self.scaler_ is not None:
            F = self.scaler_.transform(F)
        if self.pca_ is not None:
            F = self.pca_.transform(F)
        return F.astype(np.float32)

    def _build_feature_names(self) -> list[str] | None:
        """Имена после fit: при PCA компоненты теряют связь с исходными признаками —
        ['pca_000', ...]; без PCA — имена hybrid-а, при VT отфильтрованные по
        vt_.get_support()."""
        if not self._fitted:
            return None
        if self.pca_ is not None:
            return [f"pca_{i:03d}" for i in range(int(self.n_features_out_))]
        base_names = self.hybrid_.get_feature_names()
        if base_names is None:
            n_hybrid = self.hybrid_.n_features_out_
            if n_hybrid is None:
                return None
            base_names = self._generic_names(int(n_hybrid))
        if self.vt_ is not None:
            support = self.vt_.get_support()
            if len(base_names) != len(support):
                return None
            base_names = [n for n, keep in zip(base_names, support) if keep]
        return list(base_names)
