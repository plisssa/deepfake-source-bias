"""Базовый класс для всех extractor-ов признаков."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

#: меньше этого числа изображений параллелизм не окупает старт worker-пула
_PARALLEL_MIN_IMAGES = 128


def _resolve_n_jobs() -> int:
    """Число процессов экстракции: env DEEPFAKES_N_JOBS, по умолчанию -1 (все ядра)."""
    try:
        return int(os.environ.get("DEEPFAKES_N_JOBS", "-1"))
    except ValueError:
        return -1


class BaseFeatureExtractor(BaseEstimator, TransformerMixin, ABC):
    """Общий интерфейс: принимает np.ndarray путей, возвращает np.ndarray признаков.

    Экстракция параллелится по изображениям (joblib, процессы): задача
    embarrassingly parallel, а NLM/BM3D-денойзинг и Python-циклы по патчам —
    главный боттлнек всего бенчмарка. Управление: env `DEEPFAKES_N_JOBS`
    (по умолчанию -1 = все ядра) или атрибут `n_jobs` экземпляра;
    маленькие выборки (< 128) обрабатываются последовательно.
    """

    #: можно ли кэшировать выход transform (stateful-экстракторы ставят False)
    cacheable: bool = True
    #: требует ли экстрактор fit на train до вызова transform
    requires_fit: bool = False

    def __init__(self, img_size: int = 256, grayscale: bool = False, verbose: bool = True):
        self.img_size = img_size
        self.grayscale = grayscale
        self.verbose = verbose
        self.n_features_out_: int | None = None
        self.feature_names_: list[str] | None = None
        # После transform: маска успешно обработанных изображений и число отказов.
        self.last_valid_mask_: np.ndarray | None = None
        self.n_failed_: int = 0
        # Не sklearn-параметр (нет в сигнатуре __init__) — не попадает в
        # get_params и в ключ кэша признаков; на выход не влияет.
        self.n_jobs: int = _resolve_n_jobs()

    def _read_image(self, path: str) -> np.ndarray:
        if self.grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            raise ValueError(f"Can't read image: {path}")
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img

    def _prepare_raw(self, img_bgr: np.ndarray) -> np.ndarray:
        """Привести уже прочитанный BGR-кадр к входу этого экстрактора.

        Используется HybridExtractor-ом, чтобы читать каждый файл с диска один
        раз и раздавать его под-экстракторам (раньше — N чтений на файл)."""
        if self.grayscale:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

    @abstractmethod
    def _extract(self, img: np.ndarray) -> np.ndarray:
        """Извлечь признаки из одного изображения. Возвращает 1D-вектор."""

    def _safe_extract(self, path: str) -> np.ndarray | None:
        """Чтение + экстракция одного файла; None при любой ошибке (с логом)."""
        try:
            return self._extract(self._read_image(str(path)))
        except Exception as e:
            log.warning(f"  [extract failed] {path}: {e}")
            return None

    def fit(self, X, y=None):
        return self

    def get_feature_names(self) -> list[str] | None:
        """Имена признаков (len == n_features_out_) или None, если они не построены."""
        return self.feature_names_

    @staticmethod
    def _generic_names(n: int) -> list[str]:
        """Плейсхолдерные имена ['f000', 'f001', ...] — когда осмысленные не восстановить."""
        return [f"f{i:03d}" for i in range(n)]

    def _build_feature_names(self) -> list[str] | None:
        """Построить осмысленные имена признаков; None — наследник их не определяет.

        Переопределяется в конкретных экстракторах. Вызывается лениво после
        transform (см. _ensure_feature_names), поэтому наследники могут
        опираться на уже заполненный n_features_out_."""
        return None

    def _ensure_feature_names(self) -> None:
        """Лениво построить имена признаков и сверить их длину с n_features_out_.

        Несовпадение длины не роняет пайплайн: логируется warning, а имена
        сбрасываются в None (get_feature_names вернёт None)."""
        if self.feature_names_ is None:
            try:
                self.feature_names_ = self._build_feature_names()
            except Exception as e:  # имена — вспомогательная информация, не роняем
                log.warning(f"{self.__class__.__name__}: не удалось построить имена признаков: {e}")
                self.feature_names_ = None
        if (
            self.feature_names_ is not None
            and self.n_features_out_ is not None
            and len(self.feature_names_) != self.n_features_out_
        ):
            log.warning(
                f"{self.__class__.__name__}: имён признаков {len(self.feature_names_)}, "
                f"а n_features_out_={self.n_features_out_} — имена сброшены в None"
            )
            self.feature_names_ = None

    def transform(self, X) -> np.ndarray:
        n_jobs = getattr(self, "n_jobs", 1) or 1
        parallel = n_jobs not in (0, 1) and len(X) >= _PARALLEL_MIN_IMAGES
        desc = self.__class__.__name__
        if parallel:
            from joblib import Parallel, delayed

            gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(self._safe_extract)(p) for p in X
            )
            feats = list(tqdm(gen, total=len(X), desc=desc, disable=not self.verbose))
        else:
            iterator = tqdm(X, desc=desc, disable=not self.verbose)
            feats = [self._safe_extract(p) for p in iterator]
        X_out = self._finalize(feats)
        self._ensure_feature_names()
        return X_out

    def _finalize(self, feats: list[np.ndarray | None]) -> np.ndarray:
        """Собрать матрицу из списка векторов (None = невалидная строка).

        Форма выхода сохраняется (строка на каждый вход), упавшие строки — нули,
        но валидность отражена в last_valid_mask_: пайплайн выбрасывает их из
        обучения/оценки, чтобы нулевые векторы не участвовали как «настоящие»
        примеры."""
        self.last_valid_mask_ = np.array([f is not None for f in feats], dtype=bool)
        self.n_failed_ = int((~self.last_valid_mask_).sum())
        if self.n_failed_:
            log.warning(
                f"{self.__class__.__name__}: {self.n_failed_}/{len(feats)} images failed extraction"
            )
        if not self.last_valid_mask_.any():
            raise ValueError(
                f"{self.__class__.__name__}: extraction failed for ALL {len(feats)} images"
            )
        n_feat = next(f.shape[0] for f in feats if f is not None)
        feats = [f if f is not None else np.zeros(n_feat, dtype=np.float32) for f in feats]
        X_out = np.vstack(feats).astype(np.float32)
        self.n_features_out_ = X_out.shape[1]
        return X_out
