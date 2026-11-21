"""Evaluate validation-selected hyperparameters on a held-out test set."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys

import mlflow

from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger

load_project_dotenv()
log = get_logger(__name__)


def overrides_from_best_params(best: dict) -> list[str]:
    """Hydra-переопределения для параметров модели. `++` — «добавить или
    заменить»: часть ключей Optuna в yaml модели по умолчанию отсутствует."""
    out = []
    for k, v in sorted(best.items()):
        if isinstance(v, float):
            v = repr(v)  # полная точность, без округления при печати
        out.append(f"++model.params.{k}={v}")
    return out


def tuned_studies(client, experiment: str, data: str) -> list[dict]:
    exp = client.get_experiment_by_name(f"{experiment}_tune")
    if exp is None:
        return []
    out = []
    for r in client.search_runs([exp.experiment_id], max_results=1000):
        t = r.data.tags
        if r.info.status != "FINISHED" or t.get("data") != data or "best_params" not in t:
            continue
        # только протокол 2: в протоколе 1 первая попытка не была моделью
        # бенчмарка, и тюнинг мог проиграть умолчанию (col2: 0.951 против 0.968)
        if t.get("tune_protocol") != "2":
            continue
        out.append(
            {
                "pipeline": t["pipeline"],
                "model": t["model"],
                "best_params": ast.literal_eval(t["best_params"]),
                "best_val_auc": r.data.metrics.get("best_val_auc"),
            }
        )
    return out


def main() -> None:
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="gan_only")
    ap.add_argument("--cv", default="holdout")
    ap.add_argument("--experiment", default="deepfake_source_bias")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = mlflow.tracking.MlflowClient()
    studies = tuned_studies(client, args.experiment, args.data)
    if not studies:
        log.error(f"Нет завершённых исследований Optuna для data={args.data}")
        sys.exit(1)
    failed = []
    for st in studies:
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_pipeline",
            f"pipeline={st['pipeline']}",
            f"model={st['model']}",
            f"data={args.data}",
            f"cv={args.cv}",
            "seed=42",
            "tracking=mlflow_local",
            f"tracking.experiment_name={args.experiment}_tuned2",
            *overrides_from_best_params(st["best_params"]),
        ]
        log.info(
            f"{st['pipeline']}/{st['model']}: best_val_auc={st['best_val_auc']:.4f} → "
            f"переобучение и test"
        )
        log.info("  " + " ".join(cmd[2:]))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            failed.append(st["pipeline"])
            log.error(f"  упал (rc={rc})")
    if failed:
        log.error(f"Упали: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
