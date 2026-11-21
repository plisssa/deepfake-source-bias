"""Контроль «только предобработка»: детектируют ли модели DFF подделку или обработку.

Гипотеза (RESULTS §18): на DeepFakeFace модели отличают не подделанное лицо, а
предобработку датасета — все фейки DFF сжаты авторами до 512×512 и сохранены в
JPEG q75, а реальные wiki-фото лежат в исходном размере и качестве 80–100.
Прямая проверка: взять РЕАЛЬНЫЕ test-фото из wiki и пропустить их через ту же
предобработку БЕЗ всякой генерации, затем через нашу нормализацию, и подать
моделям, обученным на DFF. Если они назовут такие кадры фейками с AUC ≈ 0.99 —
механизм доказан.

Три варианта псевдо-фейков, чтобы разделить каналы:
    jpeg75          — только пересжатие JPEG q75 (канал истории сжатия);
    resize_bicubic  — ресайз до 512×512 (бикубический) + q75;
    resize_lanczos  — то же с Lanczos (точный метод ресайза датасета неизвестен).

Реальный класс — те же самые test-кадры wiki в нашей обычной нормализации,
поэтому пары «реальное — псевдо-фейк» совпадают по содержанию: различается
только обработка.

    python -m scripts.processing_control data=dm_only cv=holdout

Глубокие бейзлайны лежат в своей базе и с логрегрессией-пробой:

    python -m scripts.processing_control data=dm_only cv=holdout \
        tracking.tracking_uri=sqlite:///mlflow_gpu.db \
        '+pc.models=[logreg,xgboost]' +pc.tag=_gpu
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from sklearn.metrics import roc_auc_score

from deepfakes.eval.report import fetch_benchmark_table
from deepfakes.pipelines.base import BasePipeline
from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger
from scripts.normalize_domain import normalize_one

load_project_dotenv()
log = get_logger(__name__)

VARIANTS = {
    "jpeg75": None,
    "resize_bicubic": Image.BICUBIC,
    "resize_lanczos": Image.LANCZOS,
}


def simulate(raw: bytes, resample) -> bytes:
    """Предобработка DFF без генерации: [ресайз до 512×512] + JPEG q75 (как в датасете)."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if resample is not None:
        im = im.resize((512, 512), resample)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def original_stem(normalized_name: str) -> str:
    """'016477_1465655_1984-09-09_2014.jpg' → '1465655_1984-09-09_2014' (без индекса нормализатора)."""
    return Path(normalized_name).stem.split("_", 1)[1]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    ver = str(cfg.data.splits_version)
    test = pd.read_csv(Path(cfg.paths.splits_dir) / cfg.data.name / ver / "test.csv")
    reals = test[test.label == 0]
    log.info(f"реальных test-кадров wiki: {len(reals)}")

    zpath = Path(os.path.realpath(Path(cfg.paths.repo_root) / "data/raw/dff")) / "wiki.zip"
    zf = zipfile.ZipFile(zpath)
    by_stem = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".jpg")}

    work = Path(cfg.paths.cache_dir) / "processing_control"
    tmp = work / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    pseudo: dict[str, list[str]] = {v: [] for v in VARIANTS}
    kept_real: list[str] = []
    missing = 0
    for p in reals["path"]:
        name = Path(p).name
        entry = by_stem.get(original_stem(name))
        if entry is None:
            missing += 1
            continue
        raw = zf.read(entry)
        ok_all, outs = True, {}
        for v, rs in VARIANTS.items():
            src = tmp / f"{v}__{name}"
            src.write_bytes(simulate(raw, rs))
            dst = work / v / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            # та же нормализация, что у всего домена: кроп 256 без интерполяции,
            # JPEG q95 4:4:4, симметричный фильтр насыщенности
            res = normalize_one((str(src), str(dst), 256, 95, 12.0))
            src.unlink()
            if res != "ok":
                ok_all = False
                break
            outs[v] = str(dst)
        if ok_all:
            kept_real.append(p)
            for v in VARIANTS:
                pseudo[v].append(outs[v])
    log.info(f"псевдо-фейков на вариант: {len(kept_real)} (нет исходника: {missing})")

    runs = fetch_benchmark_table(
        experiment_name=cfg.tracking.experiment_name,
        tracking_uri=cfg.tracking.tracking_uri,
        data_name=cfg.data.name,
        cv_name=cfg.cv.name,
        splits_version=ver,
    )
    pc = cfg.get("pc", {}) or {}
    models = list(pc.get("models", ["xgboost"]))
    runs = runs[runs["tags.model"].isin(models)]
    log.info(f"моделей для проверки: {len(runs)} ({models})")
    client = mlflow.tracking.MlflowClient(tracking_uri=cfg.tracking.tracking_uri)
    rows = []
    for _, r in runs.iterrows():
        run = client.get_run(r["run_id"])
        art = Path(run.info.artifact_uri.replace("file://", "")) / "pipeline.joblib"
        if not art.exists():
            log.warning(f"{r['tags.pipeline']}: нет pipeline.joblib — пропущен")
            continue
        pipe = BasePipeline.load(art, cache_dir=cfg.paths.cache_dir)
        s_real, v_real = pipe.predict_proba(
            np.array(kept_real, dtype=object), return_valid_mask=True
        )
        row = {
            "pipeline": r["tags.pipeline"],
            "model": r["tags.model"],
            "auc_dff_test": float(r["metrics.test_roc_auc"]),
        }
        for v in VARIANTS:
            s_fake, v_fake = pipe.predict_proba(
                np.array(pseudo[v], dtype=object), return_valid_mask=True
            )
            m = v_real & v_fake
            y = np.r_[np.zeros(m.sum()), np.ones(m.sum())]
            row[f"auc_{v}"] = float(roc_auc_score(y, np.r_[s_real[m], s_fake[m]]))
            row[f"called_fake_{v}"] = float(np.mean(s_fake[m] >= pipe.threshold))
        row["called_fake_real"] = float(np.mean(s_real[v_real] >= pipe.threshold))
        rows.append(row)
        log.info(
            f"{row['pipeline']:14} {row['model']:8} DFF test {row['auc_dff_test']:.4f} | "
            + " | ".join(f"{v} {row[f'auc_{v}']:.4f}" for v in VARIANTS)
        )
    df = pd.DataFrame(rows).sort_values("auc_dff_test", ascending=False)
    name = f"processing_control_{cfg.data.name}_{ver}{pc.get('tag', '')}.csv"
    out = Path(cfg.paths.reports_dir) / "tables" / name
    df.to_csv(out, index=False)
    log.info(f"\n{df.round(4).to_string(index=False)}\nСохранено: {out}")


if __name__ == "__main__":
    main()
