"""Прогнать все пайплайны одной группы (color/freq/grad/texture/noise/ml/rob/dl).

Ключа `group` нет в базовом config.yaml, поэтому override — с плюсом:
    python -m scripts.run_group +group=color model=xgboost
    python -m scripts.run_group +group=freq  cv=k5_stratified
    python -m scripts.run_group +group=texture model=random_forest
    python -m scripts.run_group +group=all model=xgboost           # ВСЁ
"""

from __future__ import annotations

import subprocess
import sys

import hydra
from omegaconf import DictConfig

from deepfakes.pipelines.registry import REGISTRY, pipelines_by_group
from deepfakes.utils.env import load_project_dotenv

load_project_dotenv()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    group = cfg.get("group", None)
    if group is None:
        raise ValueError("Specify +group=<color|freq|grad|texture|noise|ml|rob|dl|all>")
    if group == "all":
        names = list(REGISTRY.keys())
    else:
        names = pipelines_by_group(group)
    if not names:
        print(f"Group {group} has no registered pipelines.")
        return

    cv_name = cfg.cv.name
    print(f"=== group={group}: {len(names)} pipelines × model={cfg.model.name} × cv={cv_name} ===")
    failed: list[str] = []
    for pname in names:
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_pipeline",
            f"pipeline={pname}",
            f"model={cfg.model.name}",
            f"cv={cv_name}",
            f"seed={cfg.seed}",
            f"tracking={cfg.tracking.name}",
            f"data={cfg.data.name}",
        ]
        print(">", " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            failed.append(pname)

    if failed:
        print(f"\n!!! FAILED {len(failed)}/{len(names)}: {', '.join(failed)}")
        sys.exit(1)
    print(f"\n=== group={group}: все {len(names)} пайплайнов завершились успешно ===")


if __name__ == "__main__":
    main()
