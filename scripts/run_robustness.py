"""Стенд устойчивости к пост-обработке.

Обучает пайплайн на чистом train и оценивает на test, прогнанном через набор
деградаций (JPEG-пережатие, ресайз туда-обратно, размытие, шум). Порог берётся
с чистого val и НЕ подстраивается под деградацию — иначе устойчивость окажется
завышенной.

Зачем. Литература (SAFE 2024, FreqNet AAAI 2024) предсказывает: частотные
признаки теряют качество быстрее текстурных, потому что пост-обработка бьёт
именно по высоким частотам. Наш отчёт заявлял устойчивость, не измерив её.

Запуск:
    python -m scripts.run_robustness pipeline=freq4 model=xgboost data=gan_only
    python -m scripts.run_robustness pipeline=tex2 model=xgboost data=gan_only \\
        '+robustness.degradations=[jpeg75,scale50]'

Подмножество деградаций задаётся списком в квадратных скобках (в кавычках, иначе
shell и Hydra спорят из-за запятой); без него берутся все сценарии DEGRADATIONS.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from deepfakes.augment.degrade import DEGRADATIONS
from deepfakes.eval.metrics import compute_report
from deepfakes.io.splits import load_split_csv
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run, pipeline_variant
from deepfakes.utils.caching import hash_inputs
from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger

load_project_dotenv()
log = get_logger(__name__)


def _materialize(
    paths: np.ndarray, name: str, out_root: Path, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Создать деградированные копии (детерминированно, с переиспользованием).

    Возвращает (пути, индексы исходных позиций). Индексы нужны потому, что
    нечитаемый файл выпадает из выборки: срез `y[:len(X)]` в этом случае сдвинул
    бы ВСЕ последующие метки на единицу — молча и с правдоподобным AUC.
    """
    if name == "clean":
        return paths, np.arange(len(paths))
    fn, params = DEGRADATIONS[name]
    key = hash_inputs([str(p) for p in paths], {"deg": name, "params": params, "seed": seed})
    out_dir = out_root / f"deg_{name}_{key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    new_paths: list[str] = []
    kept: list[int] = []
    for i, p in enumerate(paths):
        dst = out_dir / f"{i:06d}_{Path(str(p)).stem}.png"
        if not dst.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                log.warning(f"[{name}] не читается, выбрасываю из выборки: {p}")
                continue
            rng = np.random.default_rng((seed, i))
            # Каталог деградаций общий для всех пайплайнов (ключ не включает имя
            # пайплайна), поэтому параллельные job материализуют одни и те же
            # файлы. Пишем во временный и переименовываем: rename в пределах ФС
            # атомарен, и читатель никогда не увидит недописанный PNG.
            # расширение должно остаться последним: imwrite выбирает кодек по нему
            tmp = dst.with_name(f".{dst.stem}.{os.getpid()}.tmp{dst.suffix}")
            if not cv2.imwrite(str(tmp), fn(img, rng, **params)):
                raise OSError(f"не удалось записать {tmp}")
            os.replace(tmp, dst)
        new_paths.append(str(dst))
        kept.append(i)
    return np.array(new_paths, dtype=object), np.asarray(kept, dtype=int)


def reencoded_sizes(paths, quality: int = 95) -> np.ndarray:
    """Вес каждого изображения после перекодирования в JPEG заданного качества.

    Это мера СЖИМАЕМОСТИ содержимого, одинаково определённая для исходных файлов
    и деградированных копий (которые лежат в PNG без потерь). Вес файла на диске
    для этого не годится: у копий другой формат. NaN — если файл не читается."""
    out = np.full(len(paths), np.nan)
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            out[i] = float(len(buf))
    return out


def floor_auc(train_sizes, train_y, test_sizes, test_y) -> float:
    """Ранговый AUC одного веса; направление — по train (в test не подглядываем)."""
    from sklearn.metrics import roc_auc_score

    tr = ~np.isnan(train_sizes)
    te = ~np.isnan(test_sizes)
    sign = 1.0 if roc_auc_score(np.asarray(train_y)[tr], train_sizes[tr]) >= 0.5 else -1.0
    return float(roc_auc_score(np.asarray(test_y)[te], sign * test_sizes[te]))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_robust")

    rob_cfg = cfg.get("robustness", {}) or {}
    wanted = rob_cfg.get("degradations")
    if not wanted:
        names = list(DEGRADATIONS)
    elif isinstance(wanted, str):
        names = [n.strip() for n in wanted.split(",") if n.strip()]
    else:  # Hydra сам разбирает `key=[a,b]` в ListConfig
        names = [str(n).strip() for n in wanted]
    unknown = set(names) - set(DEGRADATIONS)
    if unknown:
        raise ValueError(f"неизвестные деградации: {sorted(unknown)}; есть: {list(DEGRADATIONS)}")

    splits_dir = Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version
    splits = load_split_csv(splits_dir)

    pipeline_cls = get_pipeline_class(cfg.pipeline.name)
    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))
    pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cfg.paths.cache_dir)

    run_name = f"robust__{cfg.data.name}__{pipeline_variant(cfg)}__{cfg.model.name}"
    out_dir = Path("reports/runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    deg_root = Path(cfg.paths.cache_dir) / "degraded"

    with mlflow_run(
        run_name=run_name,
        tags={
            "kind": "robustness",
            "pipeline": pipeline_variant(cfg),
            "model": cfg.model.name,
            "data": cfg.data.name,
        },
    ):
        # обучение на ЧИСТОМ train, порог — с чистого val
        log.info("Обучение на чистом train (порог с чистого val)")
        pipe.fit(
            splits["train"]["X"],
            splits["train"]["y"],
            X_val_paths=splits["val"]["X"],
            y_val=splits["val"]["y"],
        )
        threshold = float(pipe.threshold)
        mlflow.log_param("threshold_clean_val", threshold)

        # Тривиальный пол под каждой деградацией: сколько даёт ОДИН вес
        # (сжимаемость), измеренный одинаково для чистых и испорченных картинок.
        # Направление — по подвыборке train; 4000 объектов хватает для знака.
        rng = np.random.default_rng(cfg.seed)
        n_tr = len(splits["train"]["X"])
        idx = rng.choice(n_tr, size=min(4000, n_tr), replace=False)
        tr_sizes = reencoded_sizes(np.asarray(splits["train"]["X"])[idx])
        tr_y = np.asarray(splits["train"]["y"])[idx]

        rows = []
        for name in names:
            log.info(f"--- деградация: {name}")
            X_deg, kept = _materialize(splits["test"]["X"], name, deg_root, cfg.seed)
            y = np.asarray(splits["test"]["y"])[kept]
            if len(kept) < len(splits["test"]["X"]):
                log.warning(
                    f"[{name}] в выборке {len(kept)} из {len(splits['test']['X'])} "
                    "изображений (остальные не читаются)"
                )
            scores, valid = pipe.predict_proba(X_deg, return_valid_mask=True)
            if not valid.all():
                scores, y = scores[valid], y[valid]
            rep = compute_report(y, scores, threshold=threshold)
            y_all = np.asarray(splits["test"]["y"])[kept]
            floor = floor_auc(tr_sizes, tr_y, reencoded_sizes(X_deg), y_all)
            mlflow.log_metric(f"rob_{name}_floor_filesize", floor)
            mlflow.log_metric(f"rob_{name}_roc_auc", rep.roc_auc)
            mlflow.log_metric(f"rob_{name}_f1", rep.f1)
            mlflow.log_metric(f"rob_{name}_eer", rep.eer)
            rows.append(
                {
                    "degradation": name,
                    "roc_auc": round(rep.roc_auc, 4),
                    "f1": round(rep.f1, 4),
                    "eer": round(rep.eer, 4),
                    "n": rep.n,
                    "floor_filesize": round(floor, 4),
                    "lift_over_floor": round(rep.roc_auc - floor, 4),
                }
            )
            log.info(
                f"    AUC={rep.roc_auc:.4f}  F1={rep.f1:.4f}  EER={rep.eer:.4f}  "
                f"пол(вес)={floor:.4f}  прирост={rep.roc_auc - floor:+.4f}"
            )

        df = pd.DataFrame(rows)
        clean_auc = (
            float(df.loc[df.degradation == "clean", "roc_auc"].iloc[0])
            if "clean" in names
            else None
        )
        if clean_auc:
            df["auc_drop"] = (clean_auc - df["roc_auc"]).round(4)
        csv_path = out_dir / "robustness.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))
        log.info(f"\n{df.to_string(index=False)}")
        log.info(f"Сохранено: {csv_path}")


if __name__ == "__main__":
    main()
