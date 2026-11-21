"""Матрица обобщения leave-one-generator-out на нормализованных данных.

Для каждого генератора A: обучение на (real_even + A_even); затем оценка на
(real_odd + B_odd) для КАЖДОГО генератора B. Диагональ — тот же генератор на
непересекающихся людях, вне диагонали — перенос на невиданный генератор.

Как устроены данные (всё из data/raw/dm_norm, одна нормализация):

* real — wiki; половины even/odd по ID ЧЕЛОВЕКА, не по файлу;
* text2img / inpainting / insight — фейки DeepFakeFace, сделаны ИЗ wiki-фото,
  поэтому их тоже делим по ID человека: train и test не имеют общих людей ни в
  каком классе (урок утечки личностей 28.09.2026, 36% test-фейков);
* stylegan140k — у сгенерированных лиц личностей нет, делим по файлу.

**Оговорка про StyleGAN, которую нельзя опускать.** Реальный источник у него
«чужой»: StyleGAN учился на FFHQ (Flickr), а real здесь — wiki. Поэтому ячейки
со StyleGAN могут измерять «фото из Википедии против лиц в стиле FFHQ», а не
«реальное против сгенерированного». Чтобы это было видно, считается
**контроль разделимости источников**: wiki (real) против flickr140k (тоже real).
Если он высокий — StyleGAN-ячейки засорены источником, и отчёт это пишет.

Каждая ячейка сопровождается **тривиальным полом**: ранговый AUC одного веса
файла, направление которого выбрано на train-наборе A (в метки test не
подглядываем). Читать надо прирост над полом.

Запуск:
    python -m scripts.run_logo pipeline=col2 model=xgboost
    python -m scripts.run_logo pipeline=tex3 model=xgboost +logo.n_train=6000 +logo.n_test=2000
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from deepfakes.io.collect import DatasetSource
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run, pipeline_variant
from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything

load_project_dotenv()
log = get_logger(__name__)

PERSON_RX = r"/\d{6}_(\d+)_\d{4}-\d{2}-\d{2}_\d{4}\.jpg$"
#: генератор → (каталог в dm_norm, делить ли по личности)
GENERATORS: dict[str, tuple[str, bool]] = {
    "stylegan": ("stylegan140k", False),
    "text2img": ("text2img", True),
    "inpainting": ("inpainting", True),
    "insight": ("insight", True),
}
REAL = ("wiki", True)
#: второй РЕАЛЬНЫЙ источник — для контроля разделимости источников
SOURCE_CONTROL = ("flickr140k", False)


def _source(root: Path, name: str, subdir: str, label: int, by_person: bool, half: str, n: int):
    kw = dict(
        name=f"{name}_{half}",
        path=str(root / subdir),
        label=label,
        max_samples=n,
        use_subfolders=False,
        hash_split=half,
    )
    if by_person:
        kw.update(
            group_mode="regex",
            group_regex=PERSON_RX,
            group_namespace="wiki_person",
            hash_split_by="group",
        )
    else:
        kw.update(group_mode="none", hash_split_by="file")
    return DatasetSource(**kw)


def _block(root, name, subdir, label, by_person, half, n, seed):
    files, _, _ = _source(root, name, subdir, label, by_person, half, n).collect(seed=seed)
    return np.array(files, dtype=object), np.full(len(files), label, dtype=np.int8)


def filesize_floor(train_paths, train_y, test_paths, test_y) -> float:
    """Ранговый AUC одного веса файла; направление — по TRAIN."""
    s_tr = np.array([os.path.getsize(p) for p in train_paths], dtype=float)
    s_te = np.array([os.path.getsize(p) for p in test_paths], dtype=float)
    sign = 1.0 if roc_auc_score(train_y, s_tr) >= 0.5 else -1.0
    return float(roc_auc_score(test_y, sign * s_te))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    seed_everything(cfg.seed)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_logo")

    lcfg = cfg.get("logo", {}) or {}
    n_train = int(lcfg.get("n_train", 6000))  # на КАЖДЫЙ класс
    n_test = int(lcfg.get("n_test", 2000))
    root = Path(cfg.paths.repo_root) / "data" / "raw" / "dm_norm"

    variant = pipeline_variant(cfg)
    pipeline_cls = get_pipeline_class(cfg.pipeline.name)
    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))

    # блоки собираются один раз; test-наборы общие для всех строк матрицы
    real_tr = _block(root, "real", REAL[0], 0, REAL[1], "even", n_train, cfg.seed)
    real_te = _block(root, "real", REAL[0], 0, REAL[1], "odd", n_test, cfg.seed)
    fake_tr, fake_te = {}, {}
    for g, (sub, by_p) in GENERATORS.items():
        fake_tr[g] = _block(root, g, sub, 1, by_p, "even", n_train, cfg.seed)
        fake_te[g] = _block(root, g, sub, 1, by_p, "odd", n_test, cfg.seed)

    rows = []
    out_dir = Path("reports/logo")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"logo__{variant}__{cfg.model.name}__seed{cfg.seed}"
    with mlflow_run(
        run_name=run_name,
        tags={
            "kind": "logo",
            "pipeline": variant,
            "pipeline_class": cfg.pipeline.name,
            "model": cfg.model.name,
            "data": "logo_dm_norm",
            "seed": cfg.seed,
        },
    ):
        mlflow.log_params({"logo_n_train_per_class": n_train, "logo_n_test_per_class": n_test})

        # --- контроль разделимости источников: real против real -------------
        c_tr = _block(
            root, "ctrl", SOURCE_CONTROL[0], 1, SOURCE_CONTROL[1], "even", n_train, cfg.seed
        )
        c_te = _block(
            root, "ctrl", SOURCE_CONTROL[0], 1, SOURCE_CONTROL[1], "odd", n_test, cfg.seed
        )
        Xtr = np.concatenate([real_tr[0], c_tr[0]])
        ytr = np.concatenate([real_tr[1], c_tr[1]])
        Xte = np.concatenate([real_te[0], c_te[0]])
        yte = np.concatenate([real_te[1], c_te[1]])
        pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cfg.paths.cache_dir)
        pipe.fit(Xtr, ytr)
        s, v = pipe.predict_proba(Xte, return_valid_mask=True)
        src_ctrl = float(roc_auc_score(yte[v], s))
        mlflow.log_metric("source_control_wiki_vs_flickr_auc", src_ctrl)
        log.info(
            f"[{variant}] КОНТРОЛЬ ИСТОЧНИКОВ wiki против flickr (оба real): AUC={src_ctrl:.4f}"
        )

        # --- матрица ---------------------------------------------------------
        for a in GENERATORS:
            Xtr = np.concatenate([real_tr[0], fake_tr[a][0]])
            ytr = np.concatenate([real_tr[1], fake_tr[a][1]])
            pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cfg.paths.cache_dir)
            log.info(f"[{variant}] train: real + {a} ({len(Xtr)})")
            pipe.fit(Xtr, ytr)
            for b in GENERATORS:
                Xte = np.concatenate([real_te[0], fake_te[b][0]])
                yte = np.concatenate([real_te[1], fake_te[b][1]])
                s, v = pipe.predict_proba(Xte, return_valid_mask=True)
                auc = float(roc_auc_score(yte[v], s))
                floor = filesize_floor(Xtr, ytr, Xte, yte)
                rows.append(
                    {
                        "pipeline": variant,
                        "model": cfg.model.name,
                        "train": a,
                        "test": b,
                        "auc": round(auc, 4),
                        "floor_filesize": round(floor, 4),
                        "lift": round(auc - floor, 4),
                        "n_test": int(v.sum()),
                        "in_distribution": a == b,
                        "stylegan_involved": "stylegan" in (a, b),
                    }
                )
                mlflow.log_metric(f"logo__{a}__to__{b}", auc)
                mlflow.log_metric(f"floor__{a}__to__{b}", floor)
                log.info(f"    → {b:10} AUC={auc:.4f}  пол={floor:.4f}  прирост={auc - floor:+.4f}")

        df = pd.DataFrame(rows)
        df["source_control_auc"] = round(src_ctrl, 4)
        off = df[~df.in_distribution]
        clean = off[~off.stylegan_involved]
        summary = {
            "diag_mean_auc": df[df.in_distribution].auc.mean(),
            "offdiag_mean_auc": off.auc.mean(),
            "offdiag_dm_only_mean_auc": clean.auc.mean(),
            "offdiag_dm_only_mean_lift": clean.lift.mean(),
        }
        for k, val in summary.items():
            mlflow.log_metric(k, float(val))
        path = out_dir / f"logo__{variant}__{cfg.model.name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))

        mat = df.pivot(index="train", columns="test", values="auc").reindex(
            index=list(GENERATORS), columns=list(GENERATORS)
        )
        log.info(f"\nМатрица AUC (строки — обучение, столбцы — тест):\n{mat.to_string()}")
        log.info(
            f"Среднее: диагональ {summary['diag_mean_auc']:.4f} | вне диагонали "
            f"{summary['offdiag_mean_auc']:.4f} | вне диагонали без StyleGAN "
            f"{summary['offdiag_dm_only_mean_auc']:.4f} (прирост над полом "
            f"{summary['offdiag_dm_only_mean_lift']:+.4f})"
        )
        if src_ctrl >= 0.7:
            log.warning(
                f"Контроль источников {src_ctrl:.3f} ≥ 0.7: признаки различают wiki и "
                f"Flickr-фото БЕЗ всяких подделок. Ячейки со StyleGAN засорены "
                f"источником — выводы о переносе делать только по DM-подматрице 3×3."
            )


if __name__ == "__main__":
    main()
