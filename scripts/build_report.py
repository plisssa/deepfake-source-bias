"""Сборка финального отчёта по результатам всех MLflow runs."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from deepfakes.eval.report import (
    fetch_benchmark_table,
    fetch_negative_controls,
    write_markdown_report,
)
from deepfakes.tracking.mlflow_setup import init_mlflow
from deepfakes.utils.logging import configure_logging, get_logger


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name)

    # data_name обязателен: без него прогоны всех доменов усредняются в одну
    # ячейку (найдено 27.09.2026, см. fetch_benchmark_table)
    data_name = cfg.data.name
    # cv обязателен по той же причине: 28.09.2026 пересборка gan_only во время
    # k-fold усреднила holdout с k5_stratified в ячейках col1
    cv_name = cfg.cv.name
    tag = f"{data_name}_{cfg.data.splits_version}_{cv_name}"
    df = fetch_benchmark_table(
        experiment_name=cfg.tracking.experiment_name,
        tracking_uri=cfg.tracking.tracking_uri,
        data_name=data_name,
        cv_name=cv_name,
        splits_version=str(cfg.data.splits_version),
    )
    out_csv = Path(cfg.paths.reports_dir) / "tables" / f"benchmark_{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(f"Saved {out_csv} (n_rows={len(df)})")

    nc = fetch_negative_controls(
        experiment_name=cfg.tracking.experiment_name,
        tracking_uri=cfg.tracking.tracking_uri,
        data_name=data_name,
        splits_version=str(cfg.data.splits_version),
    )
    if len(nc):
        nc_csv = (
            Path(cfg.paths.reports_dir)
            / "tables"
            / f"negative_controls_{data_name}_{cfg.data.splits_version}.csv"
        )
        nc.to_csv(nc_csv, index=False)
        log.info(f"Saved {nc_csv}")
        res_col = "metrics.negctrl_resolution_only_auc"
        if res_col in nc.columns and float(nc[res_col].max()) >= 0.6:
            log.warning(
                f"ВНИМАНИЕ: resolution_only AUC={float(nc[res_col].max()):.3f} — "
                f"классы различимы тривиально (домен), метрики бенчмарка недостоверны "
                f"как оценка детекции дипфейков"
            )

    if len(df):
        md = Path(cfg.paths.reports_dir) / "tables" / f"benchmark_{tag}.md"
        write_markdown_report(df, md, negative_controls=nc if len(nc) else None)
        log.info(f"Saved {md}")


if __name__ == "__main__":
    main()
