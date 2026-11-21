"""ROB-2: запустить score fusion по конфигу experiment/rob2_score_fusion.yaml.

Запуск:
    python -m scripts.run_score_fusion experiment=rob2_score_fusion
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from deepfakes.eval.metrics import compute_report
from deepfakes.eval.score_fusion import (
    apply_fusion,
    collect_scores,
    fuse_logreg,
    fuse_optuna,
    fuse_uniform,
)
from deepfakes.io.splits import load_split_csv
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    seed_everything(cfg.seed)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_fusion")

    # При '- experiment: null' в defaults ключа может не быть вовсе —
    # прямое обращение cfg.experiment кидало бы ConfigAttributeError.
    if cfg.get("experiment") is None:
        raise ValueError("Specify experiment=rob2_score_fusion or another fusion config")

    splits_dir = Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version
    splits = load_split_csv(splits_dir)

    components = [
        (
            c.pipeline,
            OmegaConf.to_container(c.pipeline_cfg, resolve=True),
            OmegaConf.to_container(c.model, resolve=True),
        )
        for c in cfg.experiment.components
    ]
    names = [f"{p}__{ms['name']}" for p, _, ms in components]

    run_name = f"fusion__{cfg.experiment.mode}__{len(components)}components"
    out_dir = Path("reports/runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow_run(run_name=run_name, tags={"kind": "score_fusion", "mode": cfg.experiment.mode}):
        val_scores, test_scores, names = collect_scores(
            pipelines_and_specs=components,
            X_train=splits["train"]["X"],
            y_train=splits["train"]["y"],
            X_val=splits["val"]["X"],
            X_test=splits["test"]["X"],
            cache_dir=cfg.paths.cache_dir,
            names=names,
        )

        # Component-level reports
        per_component = {}
        for i, name in enumerate(names):
            rep = compute_report(splits["test"]["y"], test_scores[:, i])
            per_component[name] = {"test_auc": rep.roc_auc, "test_f1": rep.f1}
            mlflow.log_metric(f"comp__{name}__test_auc", rep.roc_auc)

        # Fusion
        mode = cfg.experiment.mode
        if mode == "uniform":
            result = fuse_uniform(val_scores, splits["val"]["y"], names)
        elif mode == "logreg":
            result = fuse_logreg(val_scores, splits["val"]["y"], names)
        elif mode == "optuna":
            result = fuse_optuna(
                val_scores,
                splits["val"]["y"],
                names,
                n_trials=int(cfg.experiment.optuna_n_trials),
                seed=cfg.seed,
            )
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

        log.info(f"[fusion={mode}] val_auc={result.val_auc:.4f}")
        if result.weights is not None:
            for name, w in zip(names, result.weights):
                mlflow.log_param(f"weight__{name}", float(w))
                log.info(f"  weight[{name}] = {w:.3f}")
        else:
            mlflow.log_param("meta_model", "LogisticRegression")

        # Test eval
        test_fused = apply_fusion(result, test_scores)
        test_report = compute_report(splits["test"]["y"], test_fused)
        mlflow.log_metrics(test_report.flatten_for_mlflow(prefix="fused_test_"))

        # Dump artifacts
        (out_dir / "fusion_summary.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "val_auc": result.val_auc,
                    "components": per_component,
                    "fused_test": test_report.to_dict(),
                    "weights": result.weights.tolist() if result.weights is not None else None,
                },
                indent=2,
                default=str,
            )
        )
        np.savez_compressed(
            out_dir / "scores.npz",
            val=val_scores,
            test=test_scores,
            fused_test=test_fused,
            y_val=splits["val"]["y"],
            y_test=splits["test"]["y"],
            names=np.array(names, dtype=object),
        )
        mlflow.log_artifact(str(out_dir / "fusion_summary.json"))
        mlflow.log_artifact(str(out_dir / "scores.npz"))

        log.info(
            f"FUSED TEST  acc={test_report.accuracy:.4f}  AUC={test_report.roc_auc:.4f}  F1={test_report.f1:.4f}"
        )


if __name__ == "__main__":
    main()
