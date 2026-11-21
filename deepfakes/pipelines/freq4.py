"""Radial frequency descriptors and staged classifier training."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from deepfakes.features.freq import HFEnergyExtractor, hf_gan_augment
from deepfakes.models.classical import ModelSpec, build_model
from deepfakes.pipelines.base import BasePipeline
from deepfakes.utils.caching import hash_inputs
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

StagedMode = Literal["none", "staged", "continual"]


def _lr_schedule(stage_idx: int, n_stages: int, base_lr: float, kind: str) -> float:
    """warmup_cosine: первая стадия — lr/5 (warmup), далее cosine-decay."""
    if n_stages <= 1 or kind == "none":
        return base_lr
    if kind == "warmup_cosine":
        if stage_idx == 0:
            return base_lr / 5.0
        # cosine от base_lr до base_lr/10
        t = (stage_idx - 1) / max(n_stages - 2, 1)
        return float(base_lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t))))
    if kind == "step_decay":
        return float(base_lr * (0.5**stage_idx))
    return base_lr


def _difficulty_order(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Сортировать индексы от «лёгких» к «сложным».

    Эвристика: лёгкие — те, у которых классы хорошо разделены по простому
    индикатору (среднее модуля первой главной координаты после центрирования
    относительно класса). Точное определение неважно — главное стабильный
    порядок и монотонное «усложнение» от стадии к стадии.
    """
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-8)
    mean_per_class = np.vstack([F[y == c].mean(axis=0) for c in [0, 1]])
    margin = np.linalg.norm(F[:, None, :] - mean_per_class[None, :, :], axis=2)
    # «расстояние до своего класса» — чем меньше, тем «легче»
    own = margin[np.arange(len(y)), y]
    return np.argsort(own)


def _apply_warm_start(model, lr: float, base_lr: float, stage_idx: int) -> bool:
    """Настроить модель на continual-дообучение для стадии stage_idx.

    Возвращает True, если модель реально поддерживает перенос состояния между
    стадиями (иначе caller обязан честно залогировать, что continual для неё
    вырождается в staged).
    """
    clf = model
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]

    supported = False
    if hasattr(clf, "warm_start"):
        clf.warm_start = True
        supported = True
    # MLP: lr-schedule через learning_rate_init
    if hasattr(clf, "learning_rate_init"):
        clf.learning_rate_init = lr
        # early_stopping между warm-start стадиями некорректен: состояние
        # best_score/счётчик «нет улучшений» тянется с прошлой стадии, и
        # стадии 2+ обрываются после первой итерации с откатом весов.
        if getattr(clf, "early_stopping", False):
            clf.early_stopping = False
            log.warning(
                "  [MLP] early_stopping отключён на время continual-стадий "
                "(несовместим с warm_start между стадиями)"
            )
        if hasattr(clf, "_no_improvement_count"):
            clf._no_improvement_count = 0
    # Для LogReg/SVC lr-schedule НЕ применяем: C — сила регуляризации в
    # целевой функции, а не learning rate; масштабировать его по расписанию
    # значит молча перечёркивать затюненный гиперпараметр.
    # RandomForest: warm_start без роста n_estimators НЕ добавляет деревьев —
    # наращиваем лес на каждую стадию.
    if type(clf).__name__ == "RandomForestClassifier":
        if not hasattr(clf, "_freq4_base_estimators"):
            clf._freq4_base_estimators = int(clf.n_estimators)
        clf.n_estimators = clf._freq4_base_estimators * (stage_idx + 1)
    return supported


class Freq4Pipeline(BasePipeline):
    pipeline_id = "FREQ-4"
    group = "freq"

    def __init__(
        self,
        extractor: HFEnergyExtractor,
        model_spec: ModelSpec,
        cache_dir: str | None = None,
        use_gan_augment: bool = False,
        gan_intensity: float = 0.02,
        augment_times: int = 1,
        staged_mode: StagedMode = "none",
        n_stages: int = 3,
        lr_schedule: str = "warmup_cosine",
        aug_threshold: float = 0.35,
    ):
        super().__init__(extractor=extractor, model_spec=model_spec, cache_dir=cache_dir)
        self.use_gan_augment = use_gan_augment
        self.gan_intensity = gan_intensity
        self.augment_times = augment_times
        self.staged_mode = staged_mode
        self.n_stages = int(n_stages)
        self.lr_schedule = lr_schedule
        self.aug_threshold = float(aug_threshold)

    @classmethod
    def from_cfg(cls, pipeline_cfg, model_spec: ModelSpec, cache_dir: str | None = None):
        extractor = HFEnergyExtractor(
            img_size=pipeline_cfg.get("img_size", 256),
            n_bins=pipeline_cfg.get("n_bins", 8),
            include_slope=pipeline_cfg.get("include_slope", True),
            include_flatness=pipeline_cfg.get("include_flatness", True),
            include_basic_stats=pipeline_cfg.get("include_basic_stats", True),
        )
        return cls(
            extractor=extractor,
            model_spec=model_spec,
            cache_dir=cache_dir,
            use_gan_augment=pipeline_cfg.get("use_gan_augment", False),
            gan_intensity=pipeline_cfg.get("gan_intensity", 0.02),
            augment_times=pipeline_cfg.get("augment_times", 1),
            staged_mode=pipeline_cfg.get("staged_mode", "none"),
            n_stages=pipeline_cfg.get("n_stages", 3),
            lr_schedule=pipeline_cfg.get("lr_schedule", "warmup_cosine"),
            aug_threshold=pipeline_cfg.get("aug_threshold", 0.35),
        )

    # ---------- augmentation ----------
    def augment_paths(self, X_paths: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """HF-GAN аугментация train-выборки (robustness-аугментация: обеим
        классам добавляются HF-возмущённые копии с СОХРАНЕНИЕМ метки, чтобы
        модель не завязывалась на сам факт HF-возмущения).

        Аугментация ДЕТЕРМИНИРОВАНА (hf_gan_augment без случайности):
        директория и имена файлов — функции от (пути, intensity, threshold),
        поэтому повторный запуск переиспользует уже сгенерированные файлы,
        и кэш признаков работает.
        """
        if not self.use_gan_augment or self.augment_times <= 0:
            return X_paths, y
        if self.augment_times > 1:
            log.warning(
                "[FREQ-4] augment_times>1 при детерминированной аугментации "
                "даёт идентичные копии — эффективно это augment_times=1"
            )
        key = hash_inputs(
            [str(p) for p in X_paths],
            {
                "intensity": self.gan_intensity,
                "threshold": self.aug_threshold,
                "times": self.augment_times,
                "img_size": self.extractor.img_size,
            },
        )
        base_dir = Path(self.cache_dir) if self.cache_dir else Path(tempfile.gettempdir())
        tmpdir = base_dir / f"freq4_aug_{key}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        new_paths: list[str] = []
        new_y: list[int] = []
        for i, (p, label) in enumerate(zip(X_paths, y)):
            for k in range(self.augment_times):
                out_p = tmpdir / f"{i:06d}_{k}_{Path(str(p)).stem}.png"
                if not out_p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(
                        img,
                        (self.extractor.img_size, self.extractor.img_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    aug = hf_gan_augment(
                        img, intensity=self.gan_intensity, threshold=self.aug_threshold
                    )
                    cv2.imwrite(str(out_p), aug)
                new_paths.append(str(out_p))
                new_y.append(int(label))
        X_aug = np.concatenate([X_paths, np.array(new_paths, dtype=object)])
        y_aug = np.concatenate([y, np.array(new_y, dtype=y.dtype)])
        log.info(
            f"[FREQ-4] augmented +{len(new_paths)} (intensity={self.gan_intensity}, dir={tmpdir.name})"
        )
        return X_aug, y_aug

    # ---------- staged fit ----------
    def _staged_fit(self, X: np.ndarray, y: np.ndarray):
        """Curriculum learning: easy → hard, перенос весов между стадиями.

        Для каждой стадии s ∈ [0..n_stages):
          - подмножество = первые ((s+1)/n_stages) индексов из difficulty_order
          - lr_s = _lr_schedule(s, n_stages, base_lr, lr_schedule)
          - модель дообучается на этом подмножестве (warm_start / xgb_model)
        Возвращается единая модель, обученная continual-образом.
        """
        order = _difficulty_order(X, y)
        base_params = dict(self.model_spec.params)
        # У бустингов lr лежит в 'learning_rate' (число), у sklearn-моделей —
        # в 'learning_rate_init'; у MLP 'learning_rate' — СТРОКА ('adaptive'),
        # float() на ней падал.
        if self.model_spec.name in ("xgboost", "lightgbm", "catboost"):
            base_lr = float(base_params.get("learning_rate", 0.05))
        else:
            base_lr = float(base_params.get("learning_rate_init", 0.05))
        model = build_model(self.model_spec, n_features=X.shape[1])
        self._set_scale_pos_weight(model, y)

        # XGBoost ветка: дообучение через xgb_model
        is_xgb = type(model).__name__.startswith("XGB")
        booster = None
        warned_no_continual = False

        # Scaler внутри sklearn Pipeline фиксируем ОДИН раз на полной выборке:
        # иначе каждая стадия пере-фитит scaler на своём подмножестве, и
        # warm-start веса видят по-разному отскейленные входы.
        scaler = None
        X_all = X
        if not is_xgb and hasattr(model, "named_steps") and "scaler" in model.named_steps:
            scaler = model.named_steps["scaler"].fit(X)
            X_all = scaler.transform(X)

        for s in range(self.n_stages):
            frac = (s + 1) / self.n_stages
            n_take = max(2, int(len(order) * frac))
            idx = order[:n_take]
            X_s, y_s = X_all[idx], y[idx]
            lr_s = _lr_schedule(s, self.n_stages, base_lr, self.lr_schedule)
            log.info(
                f"  [stage {s + 1}/{self.n_stages}]  n={len(idx)}  lr={lr_s:.4g}  "
                f"mode={self.staged_mode}"
            )
            if is_xgb:
                # дообучение booster-а: используем xgb_model
                params = dict(base_params)
                params["learning_rate"] = lr_s
                from xgboost import XGBClassifier

                model = XGBClassifier(**{**model.get_params(), **params})
                if self.staged_mode == "continual" and booster is not None:
                    model.fit(X_s, y_s, xgb_model=booster)
                else:
                    model.fit(X_s, y_s)
                booster = model.get_booster()
            else:
                if self.staged_mode == "continual":
                    supported = _apply_warm_start(model, lr_s, base_lr, s)
                    if not supported and not warned_no_continual:
                        # Честность: без warm_start каждый fit переобучает модель
                        # с нуля — фактически это staged, а не continual.
                        log.warning(
                            f"  [{type(model).__name__}] не поддерживает warm_start — "
                            f"continual вырождается в staged (переобучение на каждой стадии)"
                        )
                        warned_no_continual = True
                if scaler is not None:
                    # scaler уже зафичен на полной выборке — дообучаем только clf
                    model.named_steps["clf"].fit(X_s, y_s)
                else:
                    model.fit(X_s, y_s)
        return model

    # ---------- model fit hook ----------
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Единственное отличие FREQ-4 от базового пайплайна — staged/continual
        обучение. Общий fit (экстракция, кэш, порог, отчёты, drop невалидных)
        живёт в BasePipeline.fit и здесь не дублируется."""
        if self.staged_mode in ("staged", "continual"):
            self.model = self._staged_fit(X, y)
        else:
            self.model = build_model(self.model_spec, n_features=X.shape[1])
            self._set_scale_pos_weight(self.model, y)
            self.model.fit(X, y)
