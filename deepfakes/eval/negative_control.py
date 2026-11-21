"""Negative-control: проверки на data leakage и тривиальные домен-различия.

1) shuffle-labels: перемешиваем метки, обучаем заново — AUC должен упасть к 0.5.
   Если нет — где-то утечка через features (например, кэш ключ от label).

2) resolution-only-control: классификация только по width/height и mean luminance.
   Если AUC > ~0.6 — значит, real и fake различимы тривиальными признаками
   до анализа дипфейк-артефактов (нужно унифицировать препроцессинг).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class NegativeControlResult:
    name: str
    auc: float
    note: str
    # AUC каждой перестановки: при идеально разделимых классах разброс между
    # перестановками шире наивной стандартной ошибки (случайный перекос меток
    # внутри кластера класса выучивается), поэтому публикуется распределение
    values: tuple[float, ...] = ()


def shuffle_labels_control(
    pipeline_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    n_repeats: int = 3,
) -> NegativeControlResult:
    """Перемешать y_train, обучить пайплайн, посчитать AUC на val.

    Повторяется n_repeats раз с разными перестановками: одно перемешивание
    даёт AUC с разбросом ~1 SE (при n=6000 это ±0.008), и единичное значение
    0.52 неотличимо от случайного жребия. Возвращается СРЕДНЕЕ по повторам."""
    aucs: list[float] = []
    for i in range(max(1, n_repeats)):
        rng = np.random.default_rng(seed + i)
        y_shuf = y_train.copy()
        rng.shuffle(y_shuf)
        pipe = pipeline_factory()
        pipe.fit(X_train, y_shuf)
        scores = pipe.predict_proba(X_val)
        aucs.append(float(roc_auc_score(y_val, scores)))
    auc = float(np.mean(aucs))
    spread = f" (повторы: {', '.join(f'{a:.3f}' for a in aucs)})" if len(aucs) > 1 else ""
    note = ("OK (AUC≈0.5)" if 0.4 <= auc <= 0.6 else "WARNING: possible leakage") + spread
    return NegativeControlResult("shuffle_labels", auc, note, tuple(aucs))


#: группы тривиальных признаков — считаются ОТДЕЛЬНО, потому что означают
#: разные проблемы: geometry = разные размеры (умрёт после унификации ресайза),
#: filesize = разная компрессия/гладкость (переживает унификацию размеров!),
#: luminance = разная съёмка/обработка источников.
_TRIVIAL_GROUPS = {"geometry": (0, 2), "luminance": (2, 4), "filesize": (4, 5)}


def _trivial_features(path: str) -> np.ndarray:
    """[width, height, mean_lum, std_lum, file_size] — только метаданные файла."""
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.full(5, np.nan, dtype=np.float32)
        h, w = img.shape
        return np.array(
            [
                float(w),
                float(h),
                float(img.mean()),
                float(img.std()),
                float(Path(path).stat().st_size),
            ],
            dtype=np.float32,
        )
    except Exception:
        # NaN, а не нули: нулевой вектор — валидная точка, которая может
        # сама по себе стать «признаком» битого файла
        return np.full(5, np.nan, dtype=np.float32)


def _fit_auc(F_train, y_train, F_val, y_val) -> float:
    """Сколько AUC тривиальный канал МОЖЕТ дать — оценка сверху, а не слабым учеником.

    Калибровка 28.09.2026 показала, что прежняя реализация (RandomForest со
    листьями до одного объекта) систематически тянет к 0.5: на одном признаке
    каждое дерево запоминает шум, и ансамбль вырождается в сглаженный kNN.
    На синтетике с известным AUC:

        истинный 0.549 → RF 0.497   (ровно случай веса файла на GAN-домене)
        истинный 0.646 → RF 0.553   (прошёл бы порог «< 0.6»!)
        истинный 0.902 → RF 0.848

    Для негативного контроля недооценка — худшее направление ошибки: он должен
    отвечать «сколько может дать тривиальный признак», и заниженный ответ
    создаёт ложную уверенность. Поэтому теперь берётся максимум по трём
    ученикам разной природы:

    * **ранговый AUC** (только для одного признака) — это Mann–Whitney, лучшее,
      что даёт любая монотонная функция признака. Направление выбирается ПО
      TRAIN, чтобы не подглядывать в метки val;
    * **логистическая регрессия** на стандартизованных признаках — линейная связь;
    * **градиентный бустинг с крупными листьями** (min_samples_leaf=200) —
      нелинейность без запоминания шума.

    Максимум по трём ученикам на одном val даёт небольшое оптимистичное смещение
    выбора — для контроля это консервативная сторона, и так и задумано."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ok_tr = ~np.isnan(F_train).any(axis=1)
    ok_va = ~np.isnan(F_val).any(axis=1)
    Ftr, ytr = F_train[ok_tr], np.asarray(y_train)[ok_tr]
    Fva, yva = F_val[ok_va], np.asarray(y_val)[ok_va]
    if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
        return float("nan")

    aucs: list[float] = []
    if Ftr.shape[1] == 1:
        sign = 1.0 if roc_auc_score(ytr, Ftr[:, 0]) >= 0.5 else -1.0
        aucs.append(float(roc_auc_score(yva, sign * Fva[:, 0])))

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Ftr, ytr)
    aucs.append(float(roc_auc_score(yva, lr.predict_proba(Fva)[:, 1])))

    gb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        class_weight="balanced",
        random_state=42,
    )
    gb.fit(Ftr, ytr)
    aucs.append(float(roc_auc_score(yva, gb.predict_proba(Fva)[:, 1])))
    return max(aucs)


def resolution_only_control(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> list[NegativeControlResult]:
    """Тривиальные признаки метаданных — общий контроль И разбивка по группам.

    ВАЖНО (урок прогона 30.07.2026): экстракторы проекта приводят изображение к
    квадрату img_size, поэтому width/height до них НЕ доходят — высокий
    resolution_only AUC доказывает лишь НАЛИЧИЕ домен-метки в метаданных, а не
    её использование пайплайнами. Но наличие такой метки почти всегда означает
    и разную ИСТОРИЮ обработки (ресэмплинг, JPEG), которая до экстракторов
    доходит. Поэтому группы считаются отдельно: `filesize` и `luminance`
    переживают унификацию размеров и остаются сигналом домена, когда
    `geometry` уже упала до 0.5.
    """
    F_train = np.vstack([_trivial_features(str(p)) for p in tqdm(X_train, desc="trivial_train")])
    F_val = np.vstack([_trivial_features(str(p)) for p in tqdm(X_val, desc="trivial_val")])

    results: list[NegativeControlResult] = []
    auc_all = _fit_auc(F_train, y_train, F_val, y_val)
    note = (
        "OK (трив. признаки не различают)" if auc_all < 0.6 else "WARNING: тривиальный domain-gap"
    )
    results.append(NegativeControlResult("resolution_only", auc_all, note))

    for name, (lo, hi) in _TRIVIAL_GROUPS.items():
        auc_g = _fit_auc(F_train[:, lo:hi], y_train, F_val[:, lo:hi], y_val)
        note_g = "OK" if auc_g < 0.6 else f"WARNING: домен различим по группе '{name}'"
        results.append(NegativeControlResult(f"trivial_{name}", auc_g, note_g))
    return results


def save_results(results: list[NegativeControlResult], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in results]).to_csv(out_path, index=False)
    return out_path
