"""Запуск всего бенчмарка: 4 пайплайна × 3 модели → единая таблица.

Не зависит от Hydra-multirun, поэтому удобен как «одна команда из README».

Пример:
    python -m scripts.run_benchmark cv=holdout
    python -m scripts.run_benchmark cv=k5_stratified
"""

from __future__ import annotations

import itertools
import subprocess
import sys

import hydra
from omegaconf import DictConfig

PIPELINES = [
    # color
    "col1",
    "col2",
    "col3",
    # freq
    "freq1",
    "freq2",
    "freq3",
    "freq4",
    "freq5",
    # grad
    "grad1",
    "grad2",
    "grad3",
    "grad4",
    # texture
    "tex1",
    "tex2",
    "tex3",
    "tex4",
    "tex5",
    "tex6",
    # noise + forensic
    "noise1",
    "noise2",
    "noise3",
    "ela",
    # ml (специальные комбинации)
    "ml6_hybrid",
    "ml7_freq_lbp",
    # rob (только rob1; rob2 — отдельный fusion-скрипт, rob9 — отдельный cross-training)
    "rob1_fusion",
]
# ML-8 и DL-пайплайны исключены из общего бенчмарка — у них своя обвязка
# (ml8: anomaly, dl: требуется torch). Запускать вручную через make pipeline.

MODELS = ["logreg", "random_forest", "xgboost"]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cv_name = cfg.cv.name
    combos = list(itertools.product(PIPELINES, MODELS))
    failed: list[str] = []
    for i, (pipe, model) in enumerate(combos, 1):
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_pipeline",
            f"pipeline={pipe}",
            f"model={model}",
            f"cv={cv_name}",
            f"seed={cfg.seed}",
            f"tracking={cfg.tracking.name}",
            f"data={cfg.data.name}",
        ]
        print("=" * 80)
        print(f"[{i}/{len(combos)}] " + " ".join(cmd))
        print("=" * 80)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            failed.append(f"{pipe}×{model}")

    # Раньше падения молча глотались (check=False без сводки) — прогон
    # «завершался успешно» при любом числе упавших конфигураций.
    if failed:
        print(f"\n!!! FAILED {len(failed)}/{len(combos)}: {', '.join(failed)}")
        sys.exit(1)
    print(f"\n=== benchmark: все {len(combos)} конфигураций завершились успешно ===")


if __name__ == "__main__":
    main()
