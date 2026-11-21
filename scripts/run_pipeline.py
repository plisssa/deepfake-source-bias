"""Главный CLI: запуск одного пайплайна (holdout или K-fold).

Примеры:
    python -m scripts.run_pipeline pipeline=col2 model=xgboost cv=holdout
    python -m scripts.run_pipeline pipeline=freq4 model=logreg cv=k5_stratified
    python -m scripts.run_pipeline --multirun pipeline=col2,freq4,grad4,tex2 model=xgboost

Hydra скомпонует config.yaml из defaults и переопределений.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from deepfakes.eval.cv import folds_to_csv, run_kfold
from deepfakes.io.collect import DatasetBundle, DatasetSource
from deepfakes.io.splits import SplitConfig, load_split_csv, make_holdout_split, save_split_csv
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run, pipeline_variant
from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything

load_project_dotenv()


def _bundle_from_data_cfg(data_cfg) -> DatasetBundle:
    def _mk(src_cfg, label: int) -> DatasetSource:
        d = OmegaConf.to_container(src_cfg, resolve=True)
        d.setdefault("label", label)
        return DatasetSource(**d)

    real = [_mk(s, 0) for s in data_cfg.real_sources]
    fake = [_mk(s, 1) for s in data_cfg.fake_sources]
    return DatasetBundle(real_sources=real, fake_sources=fake)


def _ensure_splits(cfg: DictConfig) -> dict[str, dict[str, np.ndarray]]:
    splits_dir = Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version
    if (splits_dir / "train.csv").exists():
        return load_split_csv(splits_dir)
    builder = cfg.data.get("splits_builder")
    if builder:
        raise FileNotFoundError(f"Prepare {splits_dir} with scripts.{builder} before training.")
    bundle = _bundle_from_data_cfg(cfg.data)
    X, y, sources, groups = bundle.collect(seed=cfg.seed)
    sp_cfg = SplitConfig(
        test_size=cfg.cv.test_size,
        val_size=getattr(cfg.cv, "val_size", 0.15),
        seed=cfg.seed,
        stratify=cfg.cv.stratify,
    )
    splits = make_holdout_split(X, y, sources, sp_cfg, groups=groups)
    save_split_csv(splits, splits_dir)
    return splits


def _scored(pipeline, X, y, return_valid: bool = False):
    """Скоринг + синхронный дроп нечитаемых изображений из меток."""
    scores, valid = pipeline.predict_proba(X, return_valid_mask=True)
    y = np.asarray(y)
    if not valid.all():
        scores, y = scores[valid], y[valid]
    return (scores, y, valid) if return_valid else (scores, y)


def save_test_predictions(out_dir: Path, paths, valid, y, scores, threshold: float) -> Path:
    """Поштучные предсказания на test — для доверительных интервалов и DeLong.

    До 28.09.2026 сохранялись только агрегаты и модель: чтобы посчитать
    bootstrap-CI или парный тест между пайплайнами, пришлось бы переобучать.
    Пути сохраняются вместе со скорами: по ним прогоны разных пайплайнов
    склеиваются в пары (нечитаемые изображения у разных экстракторов могут
    отличаться, и позиционное сопоставление было бы неверным)."""
    path = out_dir / "predictions_test.npz"
    np.savez_compressed(
        path,
        path=np.asarray(paths, dtype=object)[np.asarray(valid, dtype=bool)].astype(str),
        y=np.asarray(y, dtype=np.int8),
        score=np.asarray(scores, dtype=np.float64),
        threshold=np.float64(threshold),
    )
    return path


def _run_holdout(cfg: DictConfig, splits, pipeline, run_out_dir: Path) -> None:
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    artifacts = pipeline.fit(
        X_paths=train["X"],
        y=train["y"],
        X_val_paths=val["X"],
        y_val=val["y"],
        threshold_objective=cfg.get("threshold_objective", "f1"),
    )
    # MLflow: метрики и графики для каждого сплита.
    # Единый (тюненный на val) порог для всех сплитов — иначе gap-метрики
    # сравнивают accuracy/F1 при разных порогах. Нечитаемые изображения
    # выбрасываются из метрик по valid-маске (не участвуют как нулевые векторы).
    train_score, train_y = _scored(pipeline, train["X"], train["y"])
    train_report = pipeline.log_split_to_mlflow(
        "train", train_y, train_score, run_out_dir, threshold=pipeline.threshold
    )

    val_score, val_y = _scored(pipeline, val["X"], val["y"])
    val_report = pipeline.log_split_to_mlflow(
        "val", val_y, val_score, run_out_dir, threshold=pipeline.threshold
    )

    test_score, test_y, test_valid = _scored(pipeline, test["X"], test["y"], return_valid=True)
    test_report = pipeline.log_split_to_mlflow(
        "test", test_y, test_score, run_out_dir, threshold=pipeline.threshold
    )
    pred_path = save_test_predictions(
        run_out_dir, test["X"], test_valid, test_y, test_score, float(pipeline.threshold)
    )
    mlflow.log_artifact(str(pred_path))

    # Train-Val / Val-Test gaps — закрытие замечания по риску переобучения
    gaps = pipeline.log_gaps_to_mlflow(train_report, val_report, test_report)
    if gaps:
        get_logger(__name__).info("GAPS  " + "  ".join(f"{k}={v:+.4f}" for k, v in gaps.items()))

    mlflow.log_param("threshold_chosen", float(pipeline.threshold))
    mlflow.log_param("feature_dim", int(artifacts.feature_dim))

    # Персистенция обученной модели (в legacy была, при рефакторинге потерялась)
    model_path = pipeline.save(run_out_dir / "pipeline.joblib")
    mlflow.log_artifact(str(model_path))

    get_logger(__name__).info(
        f"TEST  acc={test_report.accuracy:.4f}  AUC={test_report.roc_auc:.4f}  "
        f"F1={test_report.f1:.4f}  EER={test_report.eer:.4f}"
    )


def _run_kfold(cfg: DictConfig, splits, pipeline_cls, model_spec, cache_dir, run_out_dir: Path):
    """Stratified (group-aware) K-fold: train+val вместе, test отдельно."""
    tv_X = np.concatenate([splits["train"]["X"], splits["val"]["X"]])
    tv_y = np.concatenate([splits["train"]["y"], splits["val"]["y"]])
    tv_groups = None
    if "groups" in splits["train"] and "groups" in splits["val"]:
        tv_groups = np.concatenate([splits["train"]["groups"], splits["val"]["groups"]])

    fold_thresholds: list[float] = []
    path2group = dict(zip(tv_X.tolist(), tv_groups.tolist())) if tv_groups is not None else None

    def _threshold_subsplit(X_tr, y_tr):
        """Вложенный сплит fold-train на (fit, threshold-val): порог НЕ тюнится
        на том же fold-val, которым оценивается фолд. Group-aware при группах."""
        idx = np.arange(len(y_tr))
        if path2group is not None:
            g_tr = np.array([path2group[p] for p in X_tr.tolist()], dtype=object)
            if len(np.unique(g_tr)) < len(g_tr) and len(np.unique(g_tr)) >= 4:
                from sklearn.model_selection import GroupShuffleSplit

                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
                fit_idx, thr_idx = next(gss.split(idx, y_tr, groups=g_tr))
                return fit_idx, thr_idx
        from sklearn.model_selection import train_test_split

        return train_test_split(idx, test_size=0.2, stratify=y_tr, random_state=cfg.seed)

    def fit_predict(X_tr, y_tr, X_va, y_va):
        pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cache_dir)
        fit_idx, thr_idx = _threshold_subsplit(X_tr, y_tr)
        pipe.fit(X_tr[fit_idx], y_tr[fit_idx], X_val_paths=X_tr[thr_idx], y_val=y_tr[thr_idx])
        fold_thresholds.append(float(pipe.threshold))
        # fold-val чистый: порог взят из вложенного сплита fold-train
        return pipe.predict_proba(X_va), float(pipe.threshold)

    folds, agg = run_kfold(
        tv_X, tv_y, fit_predict, n_splits=cfg.cv.n_splits, seed=cfg.seed, groups=tv_groups
    )
    folds_csv = run_out_dir / "folds.csv"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    folds_to_csv(folds, str(folds_csv))
    mlflow.log_artifact(str(folds_csv))
    mlflow.log_metrics(agg)

    # Финальная модель на train+val; test НЕ участвует ни в обучении, ни в
    # подборе порога (раньше test передавался как val — прямая утечка порога).
    final_pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cache_dir)
    final_pipe.fit(tv_X, tv_y)
    final_pipe.threshold = float(np.median(fold_thresholds)) if fold_thresholds else 0.5
    mlflow.log_param("threshold_chosen", final_pipe.threshold)
    test_score, test_y = _scored(final_pipe, splits["test"]["X"], splits["test"]["y"])
    final_pipe.log_split_to_mlflow(
        "test", test_y, test_score, run_out_dir, threshold=final_pipe.threshold
    )
    model_path = final_pipe.save(run_out_dir / "pipeline.joblib")
    mlflow.log_artifact(str(model_path))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    seed_everything(cfg.seed)

    # Resolved config
    resolved = OmegaConf.to_container(cfg, resolve=True)
    log.info("Resolved config:\n" + yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))

    # MLflow init
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name)

    # Сплиты
    splits = _ensure_splits(cfg)

    # Пайплайн
    pipeline_cls = get_pipeline_class(cfg.pipeline.name)
    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))
    cache_dir = cfg.paths.cache_dir

    # pipeline = ВАРИАНТ (имя конфига), а не класс: rob_deep_fusion и rob1_fusion
    # делят класс, и под общим тегом отчёт подменял один другим (28.09.2026).
    # data в имени: без него прогоны разных доменов писали локальные артефакты
    # в одну папку reports/runs/<run_name>, и последний затирал предыдущий.
    variant = pipeline_variant(cfg)
    run_name = f"{variant}__{cfg.data.name}__{cfg.model.name}__{cfg.cv.name}__seed{cfg.seed}"
    tags = {
        "pipeline": variant,
        "pipeline_class": cfg.pipeline.name,
        "model": cfg.model.name,
        "cv": cfg.cv.name,
        "data": cfg.data.name,
        "splits_version": cfg.data.splits_version,
        "seed": cfg.seed,
    }
    # папка — по эксперименту: иначе, например, тюнингованный прогон (эксперимент
    # deepfake_source_bias_tuned) лёг бы в ту же папку, что базовый с тем же run_name
    run_out_dir = Path("reports/runs") / cfg.tracking.experiment_name / run_name
    run_out_dir.mkdir(parents=True, exist_ok=True)
    # save resolved config locally too
    (run_out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    with mlflow_run(run_name=run_name, tags=tags):
        mlflow.log_params(_flatten_params(resolved))
        mlflow.log_artifact(str(run_out_dir / "resolved_config.yaml"))

        if cfg.cv.type == "holdout":
            pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cache_dir)
            _run_holdout(cfg, splits, pipe, run_out_dir)
        elif cfg.cv.type == "kfold":
            _run_kfold(cfg, splits, pipeline_cls, model_spec, cache_dir, run_out_dir)
        else:
            raise ValueError(f"Unknown cv.type={cfg.cv.type}")


def _flatten_params(d, prefix: str = "", out: dict | None = None) -> dict:
    """Hydra→MLflow: разворачивает вложенный dict для log_params."""
    if out is None:
        out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            _flatten_params(v, key, out)
        elif isinstance(v, (list, tuple)):
            out[key] = str(v)
        else:
            out[key] = v
    return out


if __name__ == "__main__":
    main()
