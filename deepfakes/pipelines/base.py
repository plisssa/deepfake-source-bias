"""BasePipeline: единая обвязка для всех пайплайнов.

Каждый пайплайн = (extractor → optional augmentation → model). Этот класс берёт
на себя кэширование признаков, обучение, evaluation, MLflow-логирование, CV.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mlflow
import numpy as np
from sklearn.base import BaseEstimator

from deepfakes.eval.metrics import (
    ClassificationReport,
    compute_report,
    plot_calibration,
    plot_confusion_matrix,
    plot_pr,
    plot_roc,
    tune_threshold,
)
from deepfakes.features.base import BaseFeatureExtractor
from deepfakes.models.classical import ModelSpec, build_model, predict_proba_safe
from deepfakes.utils.caching import feature_cache_path, hash_inputs, load_cached, save_cached
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


def mlflow_safe_log_metric(name: str, value: float) -> None:
    """log_metric только при активном run — пайплайн можно вызывать и без MLflow."""
    try:
        if mlflow.active_run() is not None:
            mlflow.log_metric(name, value)
    except Exception:  # метрика не должна ронять обучение
        pass


@dataclass
class PipelineArtifacts:
    """То, что пайплайн откладывает после run-а."""

    extractor: BaseFeatureExtractor
    model: BaseEstimator
    feature_dim: int
    threshold: float
    train_report: ClassificationReport | None = None
    val_report: ClassificationReport | None = None
    test_report: ClassificationReport | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BasePipeline:
    """Общая обвязка: extract → fit → eval → MLflow.

    Подклассы могут переопределить hook-и `before_fit`, `after_fit`, `augment_paths`.
    """

    pipeline_id: str = "BASE"

    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        model_spec: ModelSpec,
        cache_dir: str | None = None,
        use_cache: bool = True,
        augment_cfg: dict | None = None,
    ):
        self.extractor = extractor
        self.model_spec = model_spec
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None
        self.augment_cfg = augment_cfg
        self.model: BaseEstimator | None = None
        self.threshold: float = 0.5
        self.extractor_params_: dict[str, Any] | None = None

    # ---------- hook-и для подклассов ----------

    def augment_paths(self, X_paths: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Универсальная аугментация train-выборки (восстановление legacy-семантики).

        Если augment_cfg задан ({'name': ..., 'times': ..., 'params': {...},
        'seed': ...}) — для каждого train-пути создаётся `times` аугментированных
        копий с СОХРАНЕНИЕМ метки (robustness-аугментация: модель не должна
        завязываться на сам факт возмущения). Файлы пишутся в детерминированную
        директорию <cache_dir|tempdir>/aug_<name>_<hash>: хэш — функция от путей,
        имени и параметров, поэтому повторный запуск переиспользует уже
        сгенерированные файлы, и кэш признаков работает. Воспроизводимость —
        rng = default_rng((seed, индекс файла, k)).
        """
        if not self.augment_cfg:
            return X_paths, y
        from deepfakes.augment import AUGMENTERS

        name = self.augment_cfg["name"]
        times = int(self.augment_cfg.get("times", 1))
        params = dict(self.augment_cfg.get("params", {}))
        seed = int(self.augment_cfg.get("seed", 0))
        if times <= 0:
            return X_paths, y
        if name not in AUGMENTERS:
            raise ValueError(f"Unknown augmenter: {name}. Known: {sorted(AUGMENTERS)}")
        augmenter = AUGMENTERS[name]

        key = hash_inputs(
            [str(p) for p in X_paths],
            {"augmenter": name, "times": times, "params": params, "seed": seed},
        )
        base_dir = Path(self.cache_dir) if self.cache_dir else Path(tempfile.gettempdir())
        tmpdir = base_dir / f"aug_{name}_{key}"
        tmpdir.mkdir(parents=True, exist_ok=True)

        new_paths: list[str] = []
        new_y: list[int] = []
        for i, (p, label) in enumerate(zip(X_paths, y)):
            for k in range(times):
                out_p = tmpdir / f"{i:06d}_{k}_{Path(str(p)).stem}.png"
                if not out_p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    rng = np.random.default_rng((seed, i, k))
                    aug = augmenter(img, rng, **params)
                    cv2.imwrite(str(out_p), aug)
                new_paths.append(str(out_p))
                new_y.append(int(label))
        y = np.asarray(y)
        X_aug = np.concatenate(
            [np.asarray(X_paths, dtype=object), np.array(new_paths, dtype=object)]
        )
        y_aug = np.concatenate([y, np.array(new_y, dtype=y.dtype)])
        log.info(f"[{self.pipeline_id}] augmented +{len(new_paths)} ({name}, dir={tmpdir.name})")
        return X_aug, y_aug

    def before_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return None

    def after_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return None

    # ---------- основная логика ----------

    @staticmethod
    def _serialize_params(obj: Any) -> Any:
        """Рекурсивная сериализация параметров для ключа кэша.

        get_params(deep=False) для вложенных estimator-ов даёт обрезанный repr —
        два разных набора под-экстракторов могли получить один ключ. Здесь
        каждый BaseEstimator разворачивается в {class, params} рекурсивно.
        """
        if isinstance(obj, BaseEstimator):
            return {
                "__class__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "params": {
                    k: BasePipeline._serialize_params(v)
                    for k, v in obj.get_params(deep=False).items()
                },
            }
        if isinstance(obj, dict):
            return {k: BasePipeline._serialize_params(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BasePipeline._serialize_params(v) for v in obj]
        return obj

    def _extract_with_cache(self, X_paths: np.ndarray, tag: str) -> np.ndarray:
        """Извлечь признаки, при возможности через npz-кэш.

        Побочный эффект: self._last_valid_mask — маска успешно обработанных
        строк (для кэш-хита читается из npz)."""
        cacheable = getattr(self.extractor, "cacheable", True)
        if not self.use_cache or not cacheable:
            X = self.extractor.transform(X_paths)
            self._last_valid_mask = getattr(self.extractor, "last_valid_mask_", None)
            if self._last_valid_mask is None:
                self._last_valid_mask = np.ones(X.shape[0], dtype=bool)
            return X
        # tag НЕ входит в ключ: признаки зависят только от экстрактора и файлов.
        # (Раньше tag='train'/'predict' давали разные ключи, и predict_proba
        # переэкстрактировал те же изображения заново — двойная стоимость.)
        params = {
            "extractor": self.extractor.__class__.__name__,
            "params": self._serialize_params(self.extractor.get_params(deep=False)),
        }
        paths_list = [str(p) for p in X_paths]
        key = hash_inputs(paths_list, params)
        cache_path = feature_cache_path(self.cache_dir, key)
        cached = load_cached(cache_path, expected_paths=paths_list)
        if cached is not None:
            X_cached, valid = cached
            self._last_valid_mask = valid
            return X_cached
        X = self.extractor.transform(X_paths)
        valid = getattr(self.extractor, "last_valid_mask_", None)
        if valid is None:
            valid = np.ones(X.shape[0], dtype=bool)
        self._last_valid_mask = valid
        save_cached(cache_path, X, paths=paths_list, valid_mask=valid)
        return X

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Построить и обучить модель. Подклассы (staged/anomaly) переопределяют
        ТОЛЬКО этот метод — общий fit с экстракцией/порогом/отчётами один."""
        self.model = build_model(self.model_spec, n_features=X.shape[1])
        self._set_scale_pos_weight(self.model, y)
        self.model.fit(X, y)

    @staticmethod
    def _set_scale_pos_weight(model: BaseEstimator, y: np.ndarray) -> None:
        """XGBoost — единственная модель без балансировки классов из коробки:
        выставляем scale_pos_weight = n_neg / n_pos, если параметр есть и не задан."""
        target = model
        if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
            target = model.named_steps["clf"]
        if "scale_pos_weight" not in getattr(target, "get_params", dict)():
            return
        if target.get_params().get("scale_pos_weight") is not None:
            return  # задан пользователем (в т.ч. явное 1.0) — не трогаем
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos > 0 and n_neg > 0:
            target.set_params(scale_pos_weight=n_neg / n_pos)

    def fit(
        self,
        X_paths: np.ndarray,
        y: np.ndarray,
        X_val_paths: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        threshold_objective: str = "f1",
    ) -> PipelineArtifacts:
        log.info(f"[{self.pipeline_id}] augment ...")
        X_paths_aug, y_aug = self.augment_paths(X_paths, y)

        # Stateful-экстрактор (например ROB-1 fusion) обязан обучиться на train
        # ДО любых transform — иначе scaler/PCA рискуют фититься на val/test.
        if getattr(self.extractor, "requires_fit", False):
            log.info(f"[{self.pipeline_id}] fit extractor on train ({len(X_paths_aug)})")
            self.extractor.fit(X_paths_aug, y_aug)

        log.info(f"[{self.pipeline_id}] extract train ({len(X_paths_aug)})")
        X_train = self._extract_with_cache(X_paths_aug, tag="train")
        valid = self._last_valid_mask
        if not valid.all():
            n_bad = int((~valid).sum())
            log.warning(f"[{self.pipeline_id}] dropping {n_bad} unreadable train images")
            mlflow_safe_log_metric("n_failed_train_images", float(n_bad))
            X_train, y_aug = X_train[valid], y_aug[valid]

        self.before_fit(X_train, y_aug)
        log.info(f"[{self.pipeline_id}] fit model={self.model_spec.name}")
        self._fit_model(X_train, y_aug)
        self.after_fit(X_train, y_aug)

        train_score = predict_proba_safe(self.model, X_train)

        val_report = None
        if X_val_paths is not None and y_val is not None:
            log.info(f"[{self.pipeline_id}] extract val ({len(X_val_paths)})")
            X_val = self._extract_with_cache(X_val_paths, tag="val")
            v_valid = self._last_valid_mask
            y_val_use = np.asarray(y_val)
            if not v_valid.all():
                X_val, y_val_use = X_val[v_valid], y_val_use[v_valid]
            val_score = predict_proba_safe(self.model, X_val)
            self.threshold = tune_threshold(y_val_use, val_score, objective=threshold_objective)
            val_report = compute_report(y_val_use, val_score, threshold=self.threshold)

        # train-отчёт считаем ТЕМ ЖЕ порогом, что и val — иначе gap-метрики
        # сравнивают метрики при разных порогах и «переобучение» искажено.
        train_report = compute_report(y_aug, train_score, threshold=self.threshold)

        return PipelineArtifacts(
            extractor=self.extractor,
            model=self.model,
            feature_dim=X_train.shape[1],
            threshold=self.threshold,
            train_report=train_report,
            val_report=val_report,
        )

    def predict_proba(
        self, X_paths: np.ndarray, return_valid_mask: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        X = self._extract_with_cache(X_paths, tag="predict")
        scores = predict_proba_safe(self.model, X)
        if return_valid_mask:
            return scores, self._last_valid_mask
        return scores

    def save(self, out_path: str | Path) -> Path:
        """Сохранить обученный пайплайн (extractor + model + threshold) на диск."""
        import joblib

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline_id": self.pipeline_id,
                "extractor": self.extractor,
                "model": self.model,
                "model_spec": self.model_spec,
                "threshold": self.threshold,
            },
            out_path,
        )
        return out_path

    @classmethod
    def load(cls, path: str | Path, cache_dir: str | None = None) -> BasePipeline:
        """Загрузить пайплайн, сохранённый `save()`, — пригодным для predict_proba.

        До 29.09.2026 существовал только `save()`, а он пишет СЛОВАРЬ, не объект:
        сохранённую модель нельзя было применить без ручной сборки. Это вскрылось,
        когда понадобилось пересчитать поштучные предсказания для доверительных
        интервалов без переобучения. Экземпляр собирается без вызова __init__
        подкласса (у подклассов разные сигнатуры конструктора), а все атрибуты,
        которыми пользуется предсказание, выставляются явно. cache_dir нужен,
        чтобы признаки брались из того же кэша, что при обучении."""
        import joblib

        d = joblib.load(path)
        if not isinstance(d, dict) or "extractor" not in d or "model" not in d:
            raise ValueError(f"{path}: не похоже на файл BasePipeline.save()")
        obj = cls.__new__(cls)
        BasePipeline.__init__(obj, d["extractor"], d["model_spec"], cache_dir=cache_dir)
        obj.model = d["model"]
        obj.threshold = float(d["threshold"])
        obj.pipeline_id = d.get("pipeline_id", cls.pipeline_id)
        return obj

    def evaluate(
        self, X_paths: np.ndarray, y: np.ndarray, threshold: float | None = None
    ) -> ClassificationReport:
        scores, valid = self.predict_proba(X_paths, return_valid_mask=True)
        y = np.asarray(y)
        if not valid.all():
            n_bad = int((~valid).sum())
            log.warning(f"[{self.pipeline_id}] evaluate: dropping {n_bad} unreadable images")
            mlflow_safe_log_metric("n_failed_eval_images", float(n_bad))
            scores, y = scores[valid], y[valid]
        t = self.threshold if threshold is None else threshold
        return compute_report(y, scores, threshold=t)

    # ---------- overfit / gap monitoring (закрытие замечания TEX-2) ----------

    @staticmethod
    def compute_gaps(
        train: ClassificationReport | None,
        val: ClassificationReport | None,
        test: ClassificationReport | None = None,
    ) -> dict[str, float]:
        """Train/validation and validation/test gaps for the tracking report."""
        gaps: dict[str, float] = {}
        if train is not None and val is not None:
            gaps["gap_train_val_auc"] = float(train.roc_auc - val.roc_auc)
            gaps["gap_train_val_acc"] = float(train.accuracy - val.accuracy)
            gaps["gap_train_val_f1"] = float(train.f1 - val.f1)
        if val is not None and test is not None:
            gaps["gap_val_test_auc"] = float(val.roc_auc - test.roc_auc)
            gaps["gap_val_test_acc"] = float(val.accuracy - test.accuracy)
        if train is not None and test is not None:
            gaps["gap_train_test_auc"] = float(train.roc_auc - test.roc_auc)
        return gaps

    def log_gaps_to_mlflow(
        self,
        train: ClassificationReport | None,
        val: ClassificationReport | None,
        test: ClassificationReport | None = None,
    ) -> dict[str, float]:
        gaps = self.compute_gaps(train, val, test)
        if gaps:
            mlflow.log_metrics(gaps)
        return gaps

    # ---------- MLflow ----------

    def log_split_to_mlflow(
        self,
        name: str,
        y_true: np.ndarray,
        y_score: np.ndarray,
        out_dir: str | Path,
        threshold: float | None = None,
    ) -> ClassificationReport:
        """Полная пачка артефактов для одного сплита: метрики + 4 plot-а + JSON."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t = self.threshold if threshold is None else threshold
        report = compute_report(y_true, y_score, threshold=t)
        mlflow.log_metrics(report.flatten_for_mlflow(prefix=f"{name}_"))
        cm_path = plot_confusion_matrix(
            report.confusion_matrix,
            out_dir / f"cm_{name}.png",
            title=f"Confusion matrix ({name})",
        )
        roc_path = plot_roc(y_true, y_score, out_dir / f"roc_{name}.png", title=f"ROC ({name})")
        pr_path = plot_pr(y_true, y_score, out_dir / f"pr_{name}.png", title=f"PR ({name})")
        cal_path = plot_calibration(
            y_true, y_score, out_dir / f"cal_{name}.png", title=f"Calibration ({name})"
        )
        json_path = out_dir / f"report_{name}.json"
        report.to_json(json_path)
        for p in (cm_path, roc_path, pr_path, cal_path, json_path):
            mlflow.log_artifact(str(p))
        return report
