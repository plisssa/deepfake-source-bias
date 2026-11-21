"""Cross-dataset матрица.

Запуск (требует data=cross_dataset):
    python -m scripts.run_cross_dataset pipeline=col2 model=xgboost data=cross_dataset
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from deepfakes.eval.cross_dataset import run_cross_dataset
from deepfakes.io.collect import DatasetBundle, DatasetSource
from deepfakes.models.classical import ModelSpec
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


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    seed_everything(cfg.seed)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_cross")

    if "train" not in cfg.data or "test" not in cfg.data:
        raise ValueError(
            "For cross-dataset use data=cross_dataset (has 'train' and 'test' sections)"
        )

    train_bundle = _bundle(cfg.data.train)
    test_bundle = _bundle(cfg.data.test)
    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))

    # протокол (data) обязан быть в имени: иначе cross_src_check и cross_gan_dm
    # одного пайплайна в MLflow неразличимы по имени (найдено 28.09.2026)
    run_name = f"cross__{cfg.data.name}__{pipeline_variant(cfg)}__{cfg.model.name}__seed{cfg.seed}"
    tags = {
        "pipeline": pipeline_variant(cfg),
        "model": cfg.model.name,
        "kind": "cross_dataset",
        "seed": cfg.seed,
        "data": cfg.data.name,
    }
    out_dir = Path("reports/runs") / run_name
    with mlflow_run(run_name=run_name, tags=tags):
        # Train→Test (forward)
        res_fwd = run_cross_dataset(
            train_bundle,
            test_bundle,
            pipeline_name=cfg.pipeline.name,
            pipeline_cfg=cfg.pipeline,
            model_spec=model_spec,
            cache_dir=cfg.paths.cache_dir,
            out_dir=out_dir / "forward",
            seed=cfg.seed,
        )
        # Test→Train (reverse) — для матрицы
        res_rev = run_cross_dataset(
            test_bundle,
            train_bundle,
            pipeline_name=cfg.pipeline.name,
            pipeline_cfg=cfg.pipeline,
            model_spec=model_spec,
            cache_dir=cfg.paths.cache_dir,
            out_dir=out_dir / "reverse",
            seed=cfg.seed,
        )
    log.info(f"FWD AUC={res_fwd.report.roc_auc:.4f}  |  REV AUC={res_rev.report.roc_auc:.4f}")


if __name__ == "__main__":
    main()
