"""Сборка финальных таблиц из MLflow runs → CSV + Markdown для статьи."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

CORE_METRICS = [
    "test_accuracy",
    "test_balanced_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
    "test_eer",
    "test_log_loss",
    "test_brier",
]


def filter_by_tag(runs: pd.DataFrame, tag: str, value: str | None, where: str) -> pd.DataFrame:
    """Строгий фильтр прогонов по тегу: нет тега — нет строк, а не «все подряд».

    История (28.09.2026). Первая версия фильтра по домену была написана как
    `if data_name and "tags.data" in runs.columns: ...` — и при ОТСУТСТВИИ колонки
    молча пропускалась. У прогонов негативных контролей тега data не было вовсе,
    поэтому отчёт по dm_only взял контроли GAN-домена и написал над ними
    «Контроли пройдены», хотя на diffusion-данных они не запускались ни разу.
    Фильтр, который при сбое возвращает всё, хуже отсутствия фильтра: он создаёт
    уверенность, которой нет.

    Прогоны, у которых тега нет или он пустой, отбрасываются и перечисляются
    в предупреждении — чтобы их можно было доразметить, а не потерять молча."""
    if not value:
        return runs
    col = f"tags.{tag}"
    if runs.empty:
        return runs
    if col not in runs.columns:
        log.error(
            f"[{where}] ни у одного из {len(runs)} прогонов нет тега '{tag}' — "
            f"принадлежность к {tag}={value} проверить нельзя, в отчёт НЕ берутся. "
            f"Поставьте тег при логировании (или доразметьте старые прогоны)."
        )
        return runs.iloc[0:0]
    untagged = runs[col].isna() | (runs[col].astype(str).str.strip() == "")
    if untagged.any():
        names = runs.loc[untagged, "tags.mlflow.runName"] if "tags.mlflow.runName" in runs else []
        log.error(
            f"[{where}] {int(untagged.sum())} прогонов без тега '{tag}' отброшены: "
            f"{list(names)[:5]}"
        )
    n_before = len(runs)
    out = runs[~untagged & (runs[col].astype(str) == str(value))]
    log.info(f"[{where}] {tag}={value}: осталось {len(out)} из {n_before}")
    return out


def fetch_benchmark_table(
    experiment_name: str = "deepfakes",
    tracking_uri: str | None = None,
    filter_string: str = "",
    data_name: str | None = None,
    cv_name: str | None = None,
    splits_version: str | None = None,
) -> pd.DataFrame:
    """Вытащить runs эксперимента в плоскую таблицу.

    `data_name` и `cv_name` ОБЯЗАТЕЛЬНЫ на практике: прогоны всех доменов и всех
    схем валидации живут в одном эксперименте, а сводная таблица усредняет всё,
    что попало в одну ячейку (pipeline, model).

    - 27.09.2026: `build_report` получал `data=gan_only`, но не использовал его;
      таблица была чистой лишь потому, что dm_only-прогоны появились позже.
    - 28.09.2026: фильтра по cv не было, и пересборка gan_only во время k-fold
      усреднила holdout с k5_stratified в ячейках col1."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        log.warning(f"Experiment '{experiment_name}' not found")
        return pd.DataFrame()
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string=filter_string)
    if runs.empty:
        # пустой эксперимент: search_runs возвращает DataFrame без колонок
        # run_id/start_time — индексация ниже падала бы с KeyError
        log.warning(f"Experiment '{experiment_name}' has no runs")
        return pd.DataFrame()

    runs = filter_by_tag(runs, "data", data_name, experiment_name)
    runs = filter_by_tag(runs, "cv", cv_name, experiment_name)
    # Версия сплитов тоже обязана быть в фильтре: 28.09.2026 diffusion-домен
    # пересчитывался с v2 (утечка личностей) на v3, и без этого фильтра упавшая
    # на v3 конфигурация молча показала бы в таблице своё значение с v2.
    runs = filter_by_tag(runs, "splits_version", splits_version, experiment_name)
    if runs.empty:
        log.warning(f"Для data={data_name}, cv={cv_name} нет прогонов в '{experiment_name}'")
        return pd.DataFrame()

    # Только завершённые раны: незавершённый (RUNNING после снятия job) даёт
    # строку с пустыми метриками и завышает N в таблице.
    if "status" in runs.columns:
        n_before = len(runs)
        runs = runs[runs["status"] == "FINISHED"]
        if len(runs) < n_before:
            log.warning(
                f"Отброшено {n_before - len(runs)} незавершённых ранов (status != FINISHED)"
            )

    # Дедупликация: после перезапуска упавшего прогона одни и те же
    # конфигурации попадают в таблицу дважды. Оставляем последний по времени.
    key_cols = [
        c
        for c in (
            "tags.pipeline",
            "tags.model",
            "tags.cv",
            "tags.data",
            "tags.splits_version",
            "tags.seed",
        )
        if c in runs.columns
    ]
    if key_cols and "start_time" in runs.columns:
        n_before = len(runs)
        runs = runs.sort_values("start_time").drop_duplicates(subset=key_cols, keep="last")
        if len(runs) < n_before:
            log.warning(
                f"Дедупликация: {n_before - len(runs)} повторных прогонов той же "
                f"конфигурации отброшено (оставлен последний)"
            )

    cols = ["tags.pipeline", "tags.model", "tags.cv", "tags.data", "tags.seed"]
    for m in CORE_METRICS:
        col = f"metrics.{m}"
        if col not in runs.columns:
            runs[col] = float("nan")
    return runs[
        [c for c in cols if c in runs.columns]
        + [f"metrics.{m}" for m in CORE_METRICS]
        + [c for c in ("run_id", "start_time") if c in runs.columns]
    ].copy()


def find_collapsed_cells(df: pd.DataFrame) -> list[str]:
    """Ячейки (pipeline, model), куда попало больше одного прогона.

    `pivot_table(aggfunc="mean")` усредняет такие ячейки БЕЗ предупреждения. Если
    в них попали разные домены или разные сплиты, в таблице окажется число,
    которого не давал ни один прогон. Поэтому коллизии обязаны быть видны."""
    if not {"tags.pipeline", "tags.model"} <= set(df.columns):
        return []
    sizes = df.groupby(["tags.pipeline", "tags.model"], dropna=False).size()
    out: list[str] = []
    for (pipe, model), n in sizes[sizes > 1].items():
        sub = df[(df["tags.pipeline"] == pipe) & (df["tags.model"] == model)]
        detail = []
        for c in ("tags.data", "tags.cv", "tags.seed"):
            if c in sub.columns:
                vals = sorted({str(v) for v in sub[c]})
                if len(vals) > 1:
                    detail.append(f"{c.removeprefix('tags.')}={vals}")
        out.append(
            f"{pipe}/{model}: {n} прогонов"
            + (f" — различаются {', '.join(detail)}" if detail else "")
        )
    return out


def make_pivot_table(df: pd.DataFrame, metric: str = "test_roc_auc") -> pd.DataFrame:
    """Сводная: строки = pipeline, столбцы = model, ячейка = metric."""
    col = f"metrics.{metric}"
    if col not in df.columns:
        raise ValueError(f"{col} not in dataframe")
    for c in find_collapsed_cells(df):
        log.error(f"УСРЕДНЕНИЕ РАЗНЫХ ПРОГОНОВ в одной ячейке — {c}")
    pivot = df.pivot_table(
        index="tags.pipeline",
        columns="tags.model",
        values=col,
        aggfunc="mean",
    )
    return pivot.round(4)


def fetch_negative_controls(
    experiment_name: str = "deepfakes",
    tracking_uri: str | None = None,
    data_name: str | None = None,
    splits_version: str | None = None,
) -> pd.DataFrame:
    """Метрики негативных контролей из эксперимента <name>_neg_ctrl.

    `data_name` — по той же причине, что в fetch_benchmark_table: контроли разных
    доменов лежат вместе, а контроль чужого домена ничего не говорит об этом."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(f"{experiment_name}_neg_ctrl")
    if exp is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        return pd.DataFrame()
    runs = filter_by_tag(runs, "data", data_name, f"{experiment_name}_neg_ctrl")
    # контроли версии 1 (RandomForest, занижал до 0.09) тега splits_version не
    # имеют и строгим фильтром отбрасываются — в отчёт идут только калиброванные
    runs = filter_by_tag(runs, "splits_version", splits_version, f"{experiment_name}_neg_ctrl")
    if runs.empty:
        log.warning(f"Негативных контролей для data={data_name} нет")
        return pd.DataFrame()
    # Последний контроль на (pipeline, model, data): после перекалибровки
    # 28.09.2026 в эксперименте лежат и прогоны версии 1 (RandomForest, занижал
    # до 0.09 AUC), и версии 2 — показывать оба значит показывать противоречие.
    key = [c for c in ("tags.pipeline", "tags.model", "tags.data") if c in runs.columns]
    if key and "start_time" in runs.columns:
        n_before = len(runs)
        runs = runs.sort_values("start_time").drop_duplicates(subset=key, keep="last")
        if len(runs) < n_before:
            log.info(
                f"Контроли: оставлен последний прогон на конфигурацию ({n_before} → {len(runs)})"
            )
    cols = [c for c in runs.columns if c.startswith("metrics.negctrl_")]
    keep = [
        c
        for c in ("tags.mlflow.runName", "tags.pipeline", "tags.model", "tags.negctrl_version")
        if c in runs.columns
    ]
    return runs[keep + cols].copy()


def _negative_control_section(nc: pd.DataFrame) -> list[str]:
    """Секция контролей + ГРОМКОЕ предупреждение при тривиальном domain-gap.

    Урок прогона 2026-07-30: при real=CelebA(178×218) и fake=140k(256×256)
    resolution_only дал AUC=1.0, то есть ROC-AUC≈1.0 у пайплайнов отражал
    разделимость источников, а не детекцию дипфейков. Такой результат обязан
    быть виден в отчёте рядом с таблицами, а не только в MLflow."""
    if nc.empty:
        # Отсутствие контролей — не нейтральный факт: без них таблица ниже не
        # отличима от июльской, где AUC≈1.0 измерял домен. Пишем громко.
        return [
            "\n## Negative controls\n",
            (
                "\n> ⚠️ **НЕГАТИВНЫЕ КОНТРОЛИ ДЛЯ ЭТОГО ДОМЕНА НЕ ЗАПУСКАЛИСЬ.**\n>\n"
                "> Метрики ниже нельзя предъявлять как качество детекции: не проверено, "
                "что тривиальные признаки (геометрия, яркость, вес файла) и перемешанные "
                "метки не дают сопоставимого AUC.\n"
            ),
        ]
    parts = ["\n## Negative controls\n"]
    res_col = "metrics.negctrl_resolution_only_auc"
    shuf_col = "metrics.negctrl_shuffle_labels_auc"
    warnings: list[str] = []
    if res_col in nc.columns:
        worst = float(nc[res_col].max())
        if worst >= 0.6:
            warnings.append(
                f"**resolution_only AUC = {worst:.3f}** — метаданные файлов несут метку "
                f"домена (real и fake приходят из разных пайплайнов подготовки). Сами "
                f"экстракторы разрешения не видят (вход приводится к квадрату img_size), "
                f"но разная история обработки — ресэмплинг, JPEG-сжатие — до них доходит "
                f"и может подменять собой детекцию артефактов генерации. Метрики ниже "
                f"нельзя предъявлять как качество детекции дипфейков без унификации "
                f"препроцессинга источников."
            )
    # Пофакторная разбивка: 'geometry' умирает после унификации размеров,
    # а 'filesize'/'luminance' переживают её и остаются каналом домена.
    grp_auc: dict[str, float] = {}
    for grp in ("geometry", "luminance", "filesize"):
        col = f"metrics.negctrl_trivial_{grp}_auc"
        if col in nc.columns:
            grp_auc[grp] = float(nc[col].max())
    for grp, v in grp_auc.items():
        if v < 0.6:
            continue
        msg = (
            f"**trivial_{grp} AUC = {v:.3f}** — классы различимы по группе "
            f"тривиальных признаков «{grp}»."
        )
        # Вес файла при ОДИНАКОВОЙ геометрии — отдельный случай: если размеры
        # и формат совпадают (geometry≈0.5), а вес различается, то разница идёт
        # от сжимаемости содержимого, а не от истории подготовки (так устроен
        # нормализованный diffusion-домен: одинаковый энкодер и q, но
        # сгенерированные изображения глаже). Это не утечка подготовки, но это
        # ПОТОЛОК, который тривиальный признак берёт сам по себе, — и метрики
        # пайплайнов надо читать как превышение над ним.
        if grp == "filesize" and grp_auc.get("geometry", 1.0) < 0.6:
            msg += (
                f" Геометрия классы не разделяет (AUC {grp_auc['geometry']:.3f}), значит "
                f"разница веса — от сжимаемости содержимого, а не от разной подготовки. "
                f"Это законный признак генератора, но и **нижняя планка**: пайплайн, "
                f"не превышающий {v:.3f}, по сути читает вес файла."
            )
        warnings.append(msg)
    if shuf_col in nc.columns:
        worst_s = float(nc[shuf_col].max())
        if worst_s > 0.6:
            warnings.append(
                f"**shuffle_labels AUC = {worst_s:.3f}** (ожидалось ≈0.5) — вероятна утечка меток."
            )
    if warnings:
        parts.append("\n> ⚠️ **ВНИМАНИЕ**\n>\n" + "\n>\n".join(f"> {w}" for w in warnings) + "\n")
    else:
        parts.append("\nКонтроли пройдены: тривиальные признаки не разделяют классы.\n")
    try:
        parts.append(nc.to_markdown(index=False))
    except ImportError:
        parts.append("```\n" + nc.to_string(index=False) + "\n```")
    parts.append("\n")
    return parts


def write_markdown_report(
    df: pd.DataFrame,
    out_md: str | Path,
    metrics: list[str] = ("test_roc_auc", "test_f1", "test_eer"),
    negative_controls: pd.DataFrame | None = None,
) -> Path:
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = ["# Benchmark results\n"]
    # Контроли — ПЕРЕД таблицами метрик: если домен тривиально разделим,
    # читатель должен узнать это раньше, чем увидит AUC≈1.0.
    if negative_controls is not None:
        parts += _negative_control_section(negative_controls)
    for m in metrics:
        parts.append(f"\n## {m}\n")
        pivot = make_pivot_table(df, metric=m)
        try:
            parts.append(pivot.to_markdown())
        except ImportError:  # нет tabulate — не роняем отчёт, пишем plain text
            parts.append("```\n" + pivot.to_string() + "\n```")
        parts.append("\n")
    out_md.write_text("\n".join(parts))
    return out_md
