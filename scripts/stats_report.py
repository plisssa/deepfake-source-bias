"""Доверительные интервалы и тесты значимости для таблицы бенчмарка.

Для каждого завершённого прогона нужного домена/версии сплитов/схемы берёт
поштучные предсказания на test:

* из артефакта `predictions_test.npz`, если прогон его сохранил (после 28.09);
* иначе загружает сохранённую модель `pipeline.joblib` и предсказывает test
  БЕЗ переобучения. Это заодно проверка воспроизводимости: пересчитанный AUC
  обязан совпасть с записанным в MLflow, расхождение печатается как ошибка.

Затем считает:
* bootstrap-CI (95%) для ROC-AUC каждого пайплайна;
* тест DeLong каждого пайплайна против лучшего (для одной модели), с поправкой
  Холма — «статистически неотличимые от лучшего» получают пометку;
* прирост над тривиальным полом (ранговый AUC одного веса файла на том же test).

Только ЧИТАЕТ MLflow и пишет файлы — можно запускать параллельно с задачами,
пишущими в трекер.

Запуск:
    python -m scripts.stats_report data=gan_only cv=holdout
    python -m scripts.stats_report data=dm_only cv=holdout +stats.model=xgboost
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

from deepfakes.eval.report import fetch_benchmark_table
from deepfakes.eval.stats import bootstrap_auc_ci, delong_test, holm
from deepfakes.io.splits import load_split_csv
from deepfakes.pipelines.base import BasePipeline
from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger

load_project_dotenv()
log = get_logger(__name__)

# Допуск сверки с MLflow. Пересчёт признаков на другом узле (другой BLAS, число
# потоков, порядок суммирования в деревьях) даёт расхождение AUC ~1e-6 — это не
# потеря воспроизводимости. Отчёт округляет до 4 знаков, поэтому ошибкой
# считаем то, что видно в отчёте.
REPRO_TOL = 1e-4


def _predictions_for_run(run_row, test_paths, test_y, artifacts_root: Path, cache_dir):
    """(paths, y, scores, откуда). Сначала артефакт, иначе — сохранённая модель."""
    art = artifacts_root
    npz = art / "predictions_test.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=False)
        return d["path"], d["y"], d["score"], "artifact"
    model = art / "pipeline.joblib"
    if not model.exists():
        return None
    pipe = BasePipeline.load(model, cache_dir=cache_dir)
    scores, valid = pipe.predict_proba(np.asarray(test_paths), return_valid_mask=True)
    return (
        np.asarray(test_paths, dtype=str)[valid],
        np.asarray(test_y)[valid],
        np.asarray(scores, float),
        "model",
    )


def filesize_floor(paths, y) -> float:
    """Ранговый AUC одного веса файла; направление — «фейк легче» (как на наших доменах).

    Здесь нет train, поэтому направление фиксировано заранее, а не подбирается
    по test: подбор по test завысил бы пол. Для обоих наших доменов фейки легче."""
    sizes = np.array([os.path.getsize(p) for p in paths], float)
    return float(roc_auc_score(y, -sizes))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    scfg = cfg.get("stats", {}) or {}
    model_for_tests = str(scfg.get("model", "xgboost"))
    n_boot = int(scfg.get("n_boot", 1000))
    data_name, cv_name, ver = cfg.data.name, cfg.cv.name, str(cfg.data.splits_version)

    # Несколько баз MLflow: обученные бейзлайны (GPU) пишутся в отдельную базу,
    # чтобы идти параллельно с CPU-цепочкой (один писатель на файл SQLite), а
    # сравнивать их с классикой нужно в одной таблице.
    uris = [cfg.tracking.tracking_uri] + [str(u) for u in scfg.get("extra_tracking_uris", [])]
    parts = []
    for uri in uris:
        part = fetch_benchmark_table(
            experiment_name=cfg.tracking.experiment_name,
            tracking_uri=uri,
            data_name=data_name,
            cv_name=cv_name,
            splits_version=ver,
        )
        if not part.empty:
            part["tracking_uri"] = uri
            parts.append(part)
            log.info(f"{uri}: {len(part)} прогонов")
    if not parts:
        log.error(f"Нет прогонов для data={data_name} {ver} cv={cv_name}")
        return
    runs = pd.concat(parts, ignore_index=True)
    dup = runs.duplicated(subset=["tags.pipeline", "tags.model"], keep=False)
    if dup.any():
        log.error(
            "Одна и та же конфигурация найдена в НЕСКОЛЬКИХ базах — сравнение "
            f"неоднозначно: {runs.loc[dup, ['tags.pipeline', 'tags.model', 'tracking_uri']].values.tolist()}"
        )
        raise SystemExit(2)
    split = load_split_csv(Path(cfg.paths.splits_dir) / data_name / ver)["test"]
    clients = {u: mlflow.tracking.MlflowClient(tracking_uri=u) for u in uris}

    tag = f"{data_name}_{ver}_{cv_name}"
    pred_dir = Path(cfg.paths.reports_dir) / "predictions" / tag
    pred_dir.mkdir(parents=True, exist_ok=True)

    preds: dict[tuple[str, str], tuple] = {}
    rows = []
    for _, r in runs.iterrows():
        key = (r["tags.pipeline"], r["tags.model"])
        client = clients[r["tracking_uri"]]
        art = Path(client.get_run(r["run_id"]).info.artifact_uri.replace("file://", ""))
        got = _predictions_for_run(r, split["X"], split["y"], art, cfg.paths.cache_dir)
        if got is None:
            log.error(f"{key}: нет ни predictions_test.npz, ни pipeline.joblib — пропущен")
            continue
        paths, y, s, src = got
        auc = roc_auc_score(y, s)
        logged = float(r["metrics.test_roc_auc"])
        if abs(auc - logged) > REPRO_TOL:
            log.error(
                f"{key}: ВОСПРОИЗВОДИМОСТЬ НАРУШЕНА — пересчитано {auc:.6f}, "
                f"в MLflow {logged:.6f} (источник: {src})"
            )
        np.savez_compressed(
            pred_dir / f"{key[0]}__{key[1]}.npz", path=paths.astype(str), y=y, score=s
        )
        preds[key] = (paths.astype(str), np.asarray(y), s)
        a, lo, hi = bootstrap_auc_ci(y, s, n_boot=n_boot)
        rows.append(
            {
                "pipeline": key[0],
                "model": key[1],
                "auc": round(a, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "auc_logged": round(logged, 4),
                "reproduced": abs(auc - logged) <= REPRO_TOL,
                "source": src,
            }
        )
        log.info(f"{key[0]:14} {key[1]:14} AUC {a:.4f} [{lo:.4f}, {hi:.4f}]  ({src})")

    df = pd.DataFrame(rows)

    # --- тривиальный пол на общем test ---------------------------------------
    any_paths, any_y, _ = next(iter(preds.values()))
    floor = filesize_floor(any_paths, any_y)
    df["floor_filesize"] = round(floor, 4)
    df["lift_over_floor"] = (df["auc"] - floor).round(4)

    # --- DeLong против лучшего ------------------------------------------------
    # model_for_tests='all' — сравнение ВСЕХ конфигураций (включая обученные
    # бейзлайны с их «моделями» logreg-пробник / finetune) с глобально лучшей;
    # иначе — пайплайны внутри одной модели.
    pool = df if model_for_tests == "all" else df[df.model == model_for_tests]
    sub = pool.sort_values("auc", ascending=False)
    df["delong_p_vs_best"] = np.nan
    df["delong_p_holm"] = np.nan
    if len(sub) >= 2:
        best_key = (sub.iloc[0]["pipeline"], sub.iloc[0]["model"])
        bp, by, bs = preds[best_key]
        best_idx = {p: i for i, p in enumerate(bp)}
        pv, keys = [], []
        for _, row in sub.iloc[1:].iterrows():
            key = (row["pipeline"], row["model"])
            p2, y2, s2 = preds[key]
            common = [p for p in p2 if p in best_idx]  # пары по путям, не по позиции
            i1 = np.array([best_idx[p] for p in common])
            m2 = {p: i for i, p in enumerate(p2)}
            i2 = np.array([m2[p] for p in common])
            if not np.array_equal(by[i1], y2[i2]):
                raise ValueError(
                    f"{key}: метки по одним и тем же путям не совпадают с {best_key} — "
                    f"предсказания рассинхронизированы, сравнение невозможно"
                )
            pv.append(delong_test(by[i1], bs[i1], s2[i2]).p_value)
            keys.append(key)
        adj = holm(pv)
        for (pn, mn), p_, a_ in zip(keys, pv, adj):
            m = (df.pipeline == pn) & (df.model == mn)
            df.loc[m, "delong_p_vs_best"] = p_
            df.loc[m, "delong_p_holm"] = a_
        tied = [f"{k[0]}/{k[1]}" for k, a in zip(keys, adj) if a >= 0.05]
        log.info(
            f"Лучший ({model_for_tests}): {best_key[0]}/{best_key[1]}. Статистически "
            f"неотличимы от него (Holm p ≥ 0.05): {tied if tied else 'никто'}"
        )

    out = (
        Path(cfg.paths.reports_dir)
        / "tables"
        / f"stats_{tag}{'' if model_for_tests == 'xgboost' else '_' + model_for_tests}.csv"
    )
    df.sort_values(["model", "auc"], ascending=[True, False]).to_csv(out, index=False)
    n_bad = int((~df["reproduced"]).sum())
    log.info(
        f"Сохранено: {out} | воспроизведено {len(df) - n_bad} из {len(df)} | "
        f"пол по весу файла {floor:.4f}"
    )


if __name__ == "__main__":
    main()
