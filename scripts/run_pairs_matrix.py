"""Матрица переноса между генераторами на парном наборе.

Обучение на pairs_A/train (реальные + генератор A), оценка на pairs_B/test для
каждого B. В отличие от LOGO на DFF (run_logo.py), здесь реальный класс ОДИН И
ТОТ ЖЕ для всех генераторов и прошёл тот же путь обработки, что и фейки, —
поэтому контроль разделимости источников не нужен, а вне-диагональные ячейки
впервые отвечают на вопрос «переносится ли детектор между генераторами».

К каждой ячейке — два пола (направление выбирается на train A):
    floor_filesize   — ранговый AUC веса файла;
    floor_blockiness — ранговый AUC блочности 8×8: остаточный канал «история
                       JPEG исходника», которую сгенерированные пиксели теряют.
И парная метрика для SD-генераторов (фейк сделан из того же кадра, что реальный):
    pair_win — доля пар, где фейк получил балл выше СВОЕГО реального кадра.

    python -m scripts.run_pairs_matrix pipeline=col2 model=xgboost +pairs.version=v2
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from sklearn.metrics import roc_auc_score

from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run, pipeline_variant
from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything

load_project_dotenv()
log = get_logger(__name__)

GENERATORS = ("vae", "img2img", "inpaint", "stylegan")
PAIRED = {"vae", "img2img", "inpaint"}  # фейк сделан из того же исходника, что real


def blockiness(path: str) -> float:
    """Средний перепад яркости на границах сетки 8×8 относительно перепада внутри
    блоков. >1 — видна сетка JPEG исходника."""
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    bx = np.arange(dx.shape[1]) % 8 == 7  # разность между столбцами 7 и 8 блока
    by = np.arange(dy.shape[0]) % 8 == 7
    edge = (dx[:, bx].mean() + dy[by, :].mean()) / 2
    inner = (dx[:, ~bx].mean() + dy[~by, :].mean()) / 2
    return float(edge / (inner + 1e-6))


def directed_floor(f_train, y_train, f_test, y_test) -> float:
    """Ранговый AUC одного признака; знак выбран на train (метки test не видим)."""
    sign = 1.0 if roc_auc_score(y_train, f_train) >= 0.5 else -1.0
    return float(roc_auc_score(y_test, sign * np.asarray(f_test)))


def pair_win(df: pd.DataFrame) -> float:
    """Доля пар (один group), где балл фейка выше балла его реального кадра."""
    real = df[df.label == 0].set_index("group").score
    fake = df[df.label == 1].set_index("group").score
    common = real.index.intersection(fake.index)
    if len(common) == 0:
        return float("nan")
    return float((fake[common].values > real[common].values).mean())


def load(splits_dir: Path, g: str, split: str, version: str = "v1") -> pd.DataFrame:
    return pd.read_csv(splits_dir / f"pairs_{g}" / version / f"{split}.csv")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    seed_everything(cfg.seed)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_pairs")
    splits_dir = Path(cfg.paths.splits_dir)
    variant = pipeline_variant(cfg)
    pipeline_cls = get_pipeline_class(cfg.pipeline.name)
    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))

    # версия набора: v1 — одинаковый конечный конвейер (канал истории сжатия),
    # v2 — одинаковая история обработки всех пикселей (build_pairs, 29.09.2026)
    ver = str((cfg.get("pairs", {}) or {}).get("version", "v1"))
    train = {g: load(splits_dir, g, "train", ver) for g in GENERATORS}
    test = {g: load(splits_dir, g, "test", ver) for g in GENERATORS}
    all_paths = pd.concat([*train.values(), *test.values()]).path.unique()
    feats = {p: (os.path.getsize(p), blockiness(p)) for p in all_paths}

    out_dir = Path(cfg.paths.reports_dir) / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with mlflow_run(
        run_name=f"pairs_{ver}__{variant}__{cfg.model.name}__seed{cfg.seed}",
        tags={
            "kind": "pairs_matrix",
            "pipeline": variant,
            "pipeline_class": cfg.pipeline.name,
            "model": cfg.model.name,
            "data": f"pairs_{ver}",
            "seed": cfg.seed,
        },
    ):
        for a in GENERATORS:
            tr = train[a]
            pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cfg.paths.cache_dir)
            log.info(f"[{variant}] train: real + {a} ({len(tr)})")
            pipe.fit(tr.path.to_numpy(dtype=object), tr.label.to_numpy())
            f_tr = np.array([feats[p] for p in tr.path])
            for b in GENERATORS:
                te = test[b].copy()
                s, v = pipe.predict_proba(te.path.to_numpy(dtype=object), return_valid_mask=True)
                te = te[v].assign(score=s)
                f_te = np.array([feats[p] for p in te.path])
                auc = float(roc_auc_score(te.label, te.score))
                fl_size = directed_floor(f_tr[:, 0], tr.label, f_te[:, 0], te.label)
                fl_block = directed_floor(f_tr[:, 1], tr.label, f_te[:, 1], te.label)
                row = {
                    "pipeline": variant,
                    "model": cfg.model.name,
                    "train": a,
                    "test": b,
                    "auc": round(auc, 4),
                    "floor_filesize": round(fl_size, 4),
                    "floor_blockiness": round(fl_block, 4),
                    "lift": round(auc - max(fl_size, fl_block), 4),
                    "pair_win": round(pair_win(te), 4) if b in PAIRED else float("nan"),
                    "n_test": len(te),
                    "in_distribution": a == b,
                }
                rows.append(row)
                mlflow.log_metric(f"pairs__{a}__to__{b}", auc)
                log.info(
                    f"    → {b:8} AUC={auc:.4f}  пол: вес {fl_size:.4f} блоч. {fl_block:.4f}"
                    f"  пары {row['pair_win']}"
                )
        df = pd.DataFrame(rows)
        path = out_dir / f"pairs_{ver}__{variant}__{cfg.model.name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))
        off = df[~df.in_distribution]
        mlflow.log_metric("diag_mean_auc", float(df[df.in_distribution].auc.mean()))
        mlflow.log_metric("offdiag_mean_auc", float(off.auc.mean()))
        mat = df.pivot(index="train", columns="test", values="auc").reindex(
            index=list(GENERATORS), columns=list(GENERATORS)
        )
        log.info(f"\nМатрица AUC (строки — обучение, столбцы — тест):\n{mat.to_string()}")


if __name__ == "__main__":
    main()
