"""Negative-control: shuffle labels + trivial features (resolution/luminance/size).

Запуск:
    python -m scripts.run_negative_control pipeline=col2 model=xgboost
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from deepfakes.eval.negative_control import (
    resolution_only_control,
    save_results,
    shuffle_labels_control,
)
from deepfakes.io.splits import load_split_csv
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run, pipeline_variant
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    seed_everything(cfg.seed)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_neg_ctrl")

    splits_dir = Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version
    splits = load_split_csv(splits_dir)

    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))
    pipeline_cls = get_pipeline_class(cfg.pipeline.name)

    def factory():
        return pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cfg.paths.cache_dir)

    out_dir = (
        Path("reports/runs")
        / f"negctrl__{cfg.data.name}__{pipeline_variant(cfg)}__{cfg.model.name}"
    )
    with mlflow_run(
        run_name=f"negctrl__{cfg.data.name}__{pipeline_variant(cfg)}__{cfg.model.name}",
        tags={
            "kind": "neg_ctrl",
            "pipeline": pipeline_variant(cfg),
            "model": cfg.model.name,
            # без data строгий фильтр отчёта отбросит прогон (см. filter_by_tag)
            "data": cfg.data.name,
            # версия инструмента: 2 = калиброванный контроль (28.09.2026),
            # у версии 1 (RandomForest) было занижение до 0.09 AUC
            "negctrl_version": "2",
            "splits_version": str(cfg.data.splits_version),
        },
    ):
        r1 = shuffle_labels_control(
            factory,
            splits["train"]["X"],
            splits["train"]["y"],
            splits["val"]["X"],
            splits["val"]["y"],
            seed=cfg.seed,
            n_repeats=int((cfg.get("negctrl", {}) or {}).get("n_repeats", 3)),
        )
        for i, v in enumerate(r1.values):
            mlflow.log_metric("negctrl_shuffle_labels_auc_rep", v, step=i)
        trivial = resolution_only_control(
            splits["train"]["X"],
            splits["train"]["y"],
            splits["val"]["X"],
            splits["val"]["y"],
        )
        results = [r1, *trivial]
        for r in results:
            mlflow.log_metric(f"negctrl_{r.name}_auc", r.auc)
            log.info(f"{r.name}: AUC={r.auc:.4f}  ({r.note})")
        out_csv = save_results(results, out_dir / "neg_ctrl.csv")
        mlflow.log_artifact(str(out_csv))


if __name__ == "__main__":
    main()
