"""Влияние дубликатов между сплитами на метрики — БЕЗ переобучения.

check_duplicates.py нашёл копии изображений по разные стороны сплита (DFF: 38
побайтовых + визуальные; GAN: почти-дубликаты StyleGAN — один и тот же
синтетический человек). Здесь из test убираются все изображения, у которых есть
копия (побайтовая или подтверждённая визуальная) в train или val, и AUC
пересчитывается по уже сохранённым поштучным предсказаниям. Если разница
пренебрежима — результаты стоят; если нет — нужны сплиты с группировкой по
кластерам дубликатов и пересчёт.

    python -m scripts.dedup_effect data=dm_only
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score


def leaked_test_paths(split_dir: Path, confirmed_csv: Path) -> set[str]:
    """Пути test, у которых есть копия в train/val: побайтовая (MD5) или
    подтверждённая попиксельно (из duplicates_confirmed_*.csv)."""
    frames = {s: pd.read_csv(split_dir / f"{s}.csv") for s in ("train", "val", "test")}

    def md5(p):
        return hashlib.md5(Path(p).read_bytes()).hexdigest()

    seen = {md5(p) for s in ("train", "val") for p in frames[s]["path"]}
    leaked = {p for p in frames["test"]["path"] if md5(p) in seen}
    if confirmed_csv.exists():
        c = pd.read_csv(confirmed_csv)
        for r in c.itertuples():
            sa, sb = r.splits.split(",")
            if sa == "test" and sb in ("train", "val"):
                leaked.add(r.path_a)
            if sb == "test" and sa in ("train", "val"):
                leaked.add(r.path_b)
    return leaked


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ver = str(cfg.data.splits_version)
    split_dir = Path(cfg.paths.splits_dir) / cfg.data.name / ver
    tables = Path(cfg.paths.reports_dir) / "tables"
    tag = f"{cfg.data.name}_{ver}"
    leaked = leaked_test_paths(split_dir, tables / f"duplicates_confirmed_{tag}.csv")
    n_test = len(pd.read_csv(split_dir / "test.csv"))
    print(
        f"=== {tag}: в test {n_test}, с копией в train/val — {len(leaked)} ({len(leaked) / n_test:.2%})"
    )

    pred_dir = Path(cfg.paths.reports_dir) / "predictions" / f"{tag}_{cfg.cv.name}"
    rows = []
    for f in sorted(pred_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        keep = ~np.isin(d["path"], list(leaked))
        a_all = roc_auc_score(d["y"], d["score"])
        a_dedup = roc_auc_score(d["y"][keep], d["score"][keep])
        rows.append(
            {"config": f.stem, "auc": a_all, "auc_dedup": a_dedup, "delta": a_dedup - a_all}
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"нет предсказаний в {pred_dir}")
        return
    print(f"конфигураций: {len(df)} | Δ = AUC(без утёкших) − AUC(все):")
    print(
        f"  медиана {df.delta.median():+.5f} | мин {df.delta.min():+.5f} | макс {df.delta.max():+.5f}"
    )
    print(df.sort_values("delta").head(5).round(5).to_string(index=False))
    df.to_csv(tables / f"dedup_effect_{tag}_{cfg.cv.name}.csv", index=False)


if __name__ == "__main__":
    main()
