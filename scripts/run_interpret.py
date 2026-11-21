"""Интерпретация признаков произвольного пайплайна.

kind: correlation | mutual_info | shap | pfi | anova | gini | all
('both' — алиас correlation+mutual_info, сохранён для обратной совместимости).

Для shap/pfi пайплайн обучается на train, объяснения считаются на val.
При kind=all дополнительно строится combined_ranking.csv — усреднение
нормированных рангов PFI и SHAP ('PFI+SHAP Ensemble' из обзора).

Запуск:
    python -m scripts.run_interpret pipeline=col2 +interpret.kind=correlation
    python -m scripts.run_interpret pipeline=col2 +interpret.kind=anova
    python -m scripts.run_interpret pipeline=col2 +interpret.kind=all
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

from deepfakes.interpret.explain import (
    combined_ranking_report,
    permutation_importance_plot,
    shap_summary,
)
from deepfakes.interpret.feature_selection import (
    anova_report,
    correlation_matrix_report,
    default_feature_names,
    gini_report,
    mutual_info_report,
)
from deepfakes.io.splits import load_split_csv
from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.registry import get_pipeline_class
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run, pipeline_variant
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything

VALID_KINDS = {"correlation", "mutual_info", "shap", "pfi", "anova", "gini"}

#: методы, стоимость которых растёт быстрее, чем линейно по числу признаков:
#: correlation строит матрицу d×d, mutual_info считает kNN-оценку на каждый
#: признак, pfi делает проход по каждому признаку. Их считаем по предотбору.
EXPENSIVE_KINDS = {"correlation", "mutual_info", "pfi"}
#: выше этого числа признаков дорогие методы идут по top-K, а не по всем
DEFAULT_MAX_FEATURES = 2000


def select_topk_by_anova(
    X: np.ndarray, y: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Индексы top-K признаков по ANOVA F и их F-значения.

    Зачем предотбор. У col2 вектор — 36 288 признаков; `df.corr()` на 28000×36288
    строит матрицу 36288², то есть 10.5 ГБ в float64, плюс маска и произведение
    того же размера — пик под 30 ГБ. Permutation importance делает проход по
    каждому признаку.

    Фильтр выбран ANOVA-F, потому что он считается за O(n·d) одним
    векторизованным проходом и ранжирует по той же величине, что мы и так
    отчитываем. Цена: рейтинги дорогих методов становятся **условными
    относительно предотбора** — это обязано быть написано в отчёте, иначе читатель
    решит, что PFI посчитан по всем признакам."""
    from sklearn.feature_selection import f_classif

    if y is None:
        return None, None
    with np.errstate(invalid="ignore", divide="ignore"):
        f_vals, _ = f_classif(X, y)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(f_vals)[::-1][:k]
    idx = np.sort(idx)  # порядок признаков сохраняем — так имена остаются читаемыми
    return idx, f_vals[idx]


def _parse_kinds(kind: str) -> set[str]:
    """Разобрать interpret.kind в набор методов ('both'/'all' — составные)."""
    if kind == "both":
        return {"correlation", "mutual_info"}
    if kind == "all":
        return set(VALID_KINDS)
    if kind in VALID_KINDS:
        return {kind}
    raise ValueError(f"Unknown interpret.kind={kind!r}. Valid: {sorted(VALID_KINDS)} | both | all")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    seed_everything(cfg.seed)
    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name + "_interpret")

    kind = cfg.get("interpret", {}).get("kind", "both")
    kinds = _parse_kinds(kind)
    threshold = float(cfg.get("interpret", {}).get("threshold", 0.95))
    top_k = int(cfg.get("interpret", {}).get("top_k", 30))
    max_features = int(cfg.get("interpret", {}).get("max_features", DEFAULT_MAX_FEATURES))

    splits_dir = Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version
    splits = load_split_csv(splits_dir)

    pipeline_cls = get_pipeline_class(cfg.pipeline.name)
    model_spec = ModelSpec(name=cfg.model.name, params=dict(cfg.model.get("params", {})))
    pipe = pipeline_cls.from_cfg(cfg.pipeline, model_spec, cache_dir=cfg.paths.cache_dir)

    need_train_features = kinds & {"correlation", "mutual_info", "anova", "gini"}
    need_model = kinds & {"shap", "pfi"}

    model = None
    X_val = y_val = None
    if need_model:
        # SHAP/PFI требуют ОБУЧЕННУЮ модель: полный fit пайплайна на train,
        # объяснения считаем на val (тот же холдаут, что и для порога).
        log.info("Training pipeline for model-based interpretation (shap/pfi)")
        artifacts = pipe.fit(
            splits["train"]["X"],
            splits["train"]["y"],
            X_val_paths=splits["val"]["X"],
            y_val=splits["val"]["y"],
        )
        model = artifacts.model
        X_val = pipe._extract_with_cache(splits["val"]["X"], tag="val")
        y_val = np.asarray(splits["val"]["y"])
        v_mask = pipe._last_valid_mask
        if not v_mask.all():
            X_val, y_val = X_val[v_mask], y_val[v_mask]

    X = y = None
    if need_train_features:
        # Извлекаем фичи на train без обучения модели (если fit уже был —
        # экстрактор обучен, а npz-кэш переиспользуется).
        # Stateful-экстрактор (ROB-1 fusion) требует fit(train) до transform —
        # тот же контракт, что в BasePipeline.fit.
        if getattr(pipe.extractor, "requires_fit", False) and model is None:
            log.info("Fitting stateful extractor on train (requires_fit=True)")
            pipe.extractor.fit(splits["train"]["X"], splits["train"]["y"])
        log.info("Extracting train features for interpretation")
        X = pipe._extract_with_cache(splits["train"]["X"], tag="train")
        y = np.asarray(splits["train"]["y"])
        t_mask = pipe._last_valid_mask
        if not t_mask.all():
            X, y = X[t_mask], y[t_mask]
        log.info(f"  features shape: {X.shape}")

    n_features = (X if X is not None else X_val).shape[1]
    feat_names = getattr(pipe.extractor, "feature_names_", None)
    if feat_names is None:
        feat_names = default_feature_names(n_features)

    out_dir = Path("reports/runs") / f"interpret__{cfg.data.name}__{pipeline_variant(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- предотбор для дорогих методов ---------------------------------------
    sel_idx = None
    if kinds & EXPENSIVE_KINDS and n_features > max_features:
        base_X = X if X is not None else X_val
        base_y = y if y is not None else y_val
        sel_idx, sel_f = select_topk_by_anova(base_X, base_y, max_features)
        if sel_idx is None:
            log.warning(
                f"Признаков {n_features} > {max_features}, но метки недоступны — "
                f"дорогие методы пропущены, чтобы не считать вечность"
            )
            kinds = kinds - EXPENSIVE_KINDS
        else:
            log.warning(
                f"Признаков {n_features} > max_features={max_features}: "
                f"correlation/mutual_info/pfi считаются по top-{len(sel_idx)} "
                f"по ANOVA F. Эти рейтинги УСЛОВНЫ относительно предотбора — "
                f"так и надо писать в отчёте. Дешёвые методы (anova, gini, shap) "
                f"идут по всем {n_features} признакам."
            )
            sel_path = out_dir / "expensive_methods_feature_subset.csv"
            import pandas as _pd

            _pd.DataFrame(
                {
                    "feature_index": sel_idx,
                    "feature": [feat_names[i] for i in sel_idx],
                    "anova_f": sel_f,
                }
            ).to_csv(sel_path, index=False)

    def _sub(arr, names_too=False):
        """Сузить матрицу (и имена) до предотобранных признаков."""
        if sel_idx is None or arr is None:
            return (arr, feat_names) if names_too else arr
        out = arr[:, sel_idx]
        return (out, [feat_names[i] for i in sel_idx]) if names_too else out

    run_name = f"interpret__{cfg.data.name}__{pipeline_variant(cfg)}__{kind}"
    with mlflow_run(
        run_name=run_name,
        tags={"kind": "interpret", "pipeline": pipeline_variant(cfg), "data": cfg.data.name},
    ):
        mlflow.log_param("interpret_n_features_total", n_features)
        mlflow.log_param("interpret_max_features", max_features)
        mlflow.log_param("interpret_expensive_subset", len(sel_idx) if sel_idx is not None else 0)
        artifacts_paths: list[str] = []
        if sel_idx is not None:
            artifacts_paths.append(str(out_dir / "expensive_methods_feature_subset.csv"))
        if "correlation" in kinds:
            X_c, names_c = _sub(X, names_too=True)
            corr_rep = correlation_matrix_report(X_c, names_c, out_dir, threshold=threshold)
            mlflow.log_metric("int1_n_redundant", corr_rep.n_redundant)
            artifacts_paths += [str(corr_rep.corr_matrix_path), str(corr_rep.redundant_pairs_csv)]
        if "mutual_info" in kinds:
            X_m, names_m = _sub(X, names_too=True)
            mi_rep = mutual_info_report(X_m, y, names_m, out_dir, top_k=top_k, seed=cfg.seed)
            artifacts_paths += [str(mi_rep.importance_csv), str(mi_rep.plot_path)]
        if "anova" in kinds:
            an_rep = anova_report(X, y, feat_names, out_dir, top_k=top_k)
            artifacts_paths += [str(an_rep.importance_csv), str(an_rep.plot_path)]
        if "gini" in kinds:
            gi_rep = gini_report(X, y, feat_names, out_dir, top_k=top_k, seed=cfg.seed)
            artifacts_paths += [str(gi_rep.importance_csv), str(gi_rep.plot_path)]

        shap_rep = pfi_rep = None
        if "shap" in kinds:
            shap_rep = shap_summary(
                model, X_val, out_dir / "shap_summary.png", feature_names=feat_names
            )
            if shap_rep is not None:
                artifacts_paths += [str(shap_rep.importance_csv), str(shap_rep.plot_path)]
        if "pfi" in kinds:
            # PFI делает проход по каждому признаку — здесь предотбор обязателен.
            # ВАЖНО: перестановка идёт внутри суженной матрицы, поэтому модель
            # нельзя переиспользовать как есть; permutation_importance_plot
            # получает и модель, и матрицу одной размерности только когда
            # предотбора нет. При предотборе честнее сказать это прямо.
            if sel_idx is not None:
                log.warning(
                    "PFI пропущен: модель обучена на всех "
                    f"{n_features} признаках, а перестановка по подмножеству дала бы "
                    "importances несопоставимые с этой моделью. Для PFI нужен "
                    "отдельный прогон с пайплайном меньшей размерности."
                )
                pfi_rep = None
            else:
                pfi_rep = permutation_importance_plot(
                    model,
                    X_val,
                    y_val,
                    out_dir / "permutation_importance.png",
                    feature_names=feat_names,
                    top_k=top_k,
                )
            if pfi_rep is not None:
                artifacts_paths += [str(pfi_rep.importance_csv), str(pfi_rep.plot_path)]

        if kind == "all":
            if shap_rep is not None and pfi_rep is not None:
                combined_csv = combined_ranking_report(
                    pfi_rep.importance_csv, shap_rep.importance_csv, out_dir
                )
                artifacts_paths.append(str(combined_csv))
            else:
                log.warning("Combined ranking skipped: SHAP or PFI report unavailable")

        for a in artifacts_paths:
            mlflow.log_artifact(a)
        log.info(f"Saved {len(artifacts_paths)} artifacts to {out_dir}")


if __name__ == "__main__":
    main()
