"""INT-5: LIME-объяснения для уже обученного пайплайна.

Запуск:
    python -m scripts.run_lime pipeline=tex2 model=random_forest +lime.k=5

В отличие от run_pipeline.py, этот скрипт переобучает пайплайн на train+val,
а потом делает LIME-объяснения на test (top-K fake и top-K FP).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

from deepfakes.interpret.lime_image import explain_topk
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
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_lime")

    splits_dir = Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version
    splits = load_split_csv(splits_dir)

    pipeline_cls = get_pipeline_class(cfg.pipeline.name)
    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))
    pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cfg.paths.cache_dir)

    # Обучаем на train+val
    X_tv = np.concatenate([splits["train"]["X"], splits["val"]["X"]])
    y_tv = np.concatenate([splits["train"]["y"], splits["val"]["y"]])
    log.info("Fitting pipeline on train+val for LIME analysis")
    pipe.fit(X_tv, y_tv)

    k = int(cfg.get("lime", {}).get("k", 5))
    num_samples = int(cfg.get("lime", {}).get("num_samples", 500))
    img_size = int(cfg.get("lime", {}).get("img_size", cfg.pipeline.get("img_size", 256)))

    out_dir = (
        Path("reports/runs") / f"lime__{cfg.data.name}__{pipeline_variant(cfg)}__{cfg.model.name}"
    )
    run_name = f"lime__{cfg.data.name}__{pipeline_variant(cfg)}__{cfg.model.name}"
    with mlflow_run(
        run_name=run_name,
        tags={"kind": "lime", "pipeline": cfg.pipeline.name, "model": cfg.model.name},
    ):
        paths = explain_topk(
            pipe,
            splits["test"]["X"],
            splits["test"]["y"],
            out_dir=out_dir,
            k=k,
            img_size=img_size,
            num_samples=num_samples,
        )
        for p in paths:
            mlflow.log_artifact(str(p))
        log.info(f"Saved {len(paths)} LIME explanations to {out_dir}")


if __name__ == "__main__":
    main()
