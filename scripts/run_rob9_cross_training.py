"""ROB-9: обучение на СМЕСИ нескольких датасетов, тест на out-of-distribution.

Отличается от scripts.run_cross_dataset (forward+reverse): здесь train содержит
сэмплы из обоих in-distribution bundles в заданной пропорции, а test — третий,
не виденный при обучении.

Запуск (требует data=cross_dataset с train/test секциями + extra ood_test источник):
    python -m scripts.run_rob9_cross_training pipeline=col2 model=xgboost data=cross_dataset
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from deepfakes.eval.metrics import compute_report
from deepfakes.io.collect import DatasetBundle, DatasetSource
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run, pipeline_variant
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything


def _bundle(section) -> DatasetBundle:
    def _mk(s, lbl):
        d = OmegaConf.to_container(s, resolve=True)
        d.setdefault("label", lbl)
        return DatasetSource(**d)

    real = [_mk(s, 0) for s in section.real_sources]
    fake = [_mk(s, 1) for s in section.fake_sources]
    return DatasetBundle(real_sources=real, fake_sources=fake)


def _mix(X1, y1, X2, y2, groups2, ratio: float, seed: int = 42):
    """Смешать два бандла, оставив ~`ratio` от второго (между 0 и 1).

    Отбор из B — ЦЕЛЫМИ ГРУППАМИ (видео/identity): иначе для покадрового
    датасета почти каждое видео B отдаёт часть кадров в train-микс, а часть —
    в «OOD»-тест, и OOD-метрики завышаются. Возвращает индексы использованных
    сэмплов B — именно они исключаются из OOD-теста."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups2, return_counts=True)
    order = rng.permutation(len(uniq))
    target = max(1, int(len(X2) * ratio))
    chosen, total = [], 0
    for gi in order:
        chosen.append(uniq[gi])
        total += counts[gi]
        if total >= target:
            break
    idx = np.where(np.isin(groups2, np.array(chosen, dtype=object)))[0]
    X_mix = np.concatenate([X1, X2[idx]])
    y_mix = np.concatenate([y1, y2[idx]])
    return X_mix, y_mix, idx


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    seed_everything(cfg.seed)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_rob9")

    if "train" not in cfg.data or "test" not in cfg.data:
        raise ValueError("ROB-9 requires data=cross_dataset (with train/test sections)")

    bundle_a = _bundle(cfg.data.train)
    bundle_b = _bundle(cfg.data.test)

    log.info("Collecting bundle A …")
    Xa, ya, _, _ = bundle_a.collect(seed=cfg.seed)
    log.info("Collecting bundle B …")
    Xb, yb, _, gb = bundle_b.collect(seed=cfg.seed)

    mix_ratio = float(cfg.get("rob9", {}).get("mix_ratio_b_into_a", 0.2))
    X_train, y_train, used_idx = _mix(Xa, ya, Xb, yb, gb, ratio=mix_ratio, seed=cfg.seed)
    log.info(f"  mixed train: {len(X_train)} (A={len(Xa)} + ratio={mix_ratio} of B={len(Xb)})")

    # Тест — РОВНО те сэмплы B, что не вошли в train (исключаем used_idx)
    mask = np.ones(len(Xb), dtype=bool)
    mask[used_idx] = False
    X_test, y_test = Xb[mask], yb[mask]
    assert len(set(X_train) & set(X_test)) == 0, "train/OOD-test overlap — leakage!"
    assert set(gb[used_idx]).isdisjoint(set(gb[mask])), "group leakage into OOD test!"
    log.info(f"  ood test: {len(X_test)} ({len(set(gb[mask]))} groups)")

    pipeline_cls = get_pipeline_class(cfg.pipeline.name)
    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))
    pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cfg.paths.cache_dir)

    run_name = (
        f"rob9__{cfg.data.name}__{pipeline_variant(cfg)}__{cfg.model.name}__mix{mix_ratio:.2f}"
    )
    out_dir = Path("reports/runs") / run_name
    with mlflow_run(
        run_name=run_name,
        tags={
            "kind": "rob9_cross_training",
            "pipeline": pipeline_variant(cfg),
            "model": cfg.model.name,
            "mix_ratio": mix_ratio,
            "data": cfg.data.name,
        },
    ):
        pipe.fit(X_train, y_train)
        scores, valid = pipe.predict_proba(X_test, return_valid_mask=True)
        if not valid.all():
            scores, y_test = scores[valid], y_test[valid]
        report = compute_report(y_test, scores)
        pipe.log_split_to_mlflow("ood_test", y_test, scores, out_dir)
        mlflow.log_metric("ood_test_auc", report.roc_auc)
        log.info(f"OOD test AUC={report.roc_auc:.4f}  F1={report.f1:.4f}  EER={report.eer:.4f}")


if __name__ == "__main__":
    main()
