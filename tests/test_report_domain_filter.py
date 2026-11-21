"""Отчёт не должен смешивать домены в одной ячейке.

Регрессия 27.09.2026: `build_report` получал `data=gan_only` в командной строке,
но `fetch_benchmark_table` этот параметр не принимал вовсе. Прогоны gan_only и
dm_only живут в одном MLflow-эксперименте, а `pivot_table(aggfunc="mean")`
усредняет их БЕЗ предупреждения — в таблице появилось бы число, которого не
давал ни один прогон. Таблица за 27.09 оказалась корректной только потому, что
dm_only-прогоны создались уже после её сборки.
"""

from __future__ import annotations

import pandas as pd

from deepfakes.eval.report import find_collapsed_cells, make_pivot_table


def _runs() -> pd.DataFrame:
    """Две ячейки col1: одна из gan_only, одна из dm_only — как было на кластере."""
    return pd.DataFrame(
        {
            "tags.pipeline": ["col1", "col1", "tex2"],
            "tags.model": ["xgboost", "xgboost", "xgboost"],
            "tags.data": ["gan_only", "dm_only", "gan_only"],
            "tags.cv": ["holdout"] * 3,
            "tags.seed": ["42"] * 3,
            "metrics.test_roc_auc": [0.835469, 0.741553, 0.8589],
            "status": ["FINISHED"] * 3,
        }
    )


def test_collapsed_cell_is_detected_with_reason():
    found = find_collapsed_cells(_runs())
    assert len(found) == 1
    assert "col1/xgboost" in found[0]
    assert "data=" in found[0], "должно быть видно, ЧЕМ различаются прогоны"


def test_no_collision_after_filtering_by_domain():
    gan = _runs()
    gan = gan[gan["tags.data"] == "gan_only"]
    assert find_collapsed_cells(gan) == []
    pivot = make_pivot_table(gan)
    # значение должно быть ИМЕННО прогонное, а не среднее двух домен
    assert pivot.loc["col1", "xgboost"] == 0.8355
    assert pivot.loc["col1", "xgboost"] != round((0.835469 + 0.741553) / 2, 4)


def test_unfiltered_pivot_averages_domains_and_says_so():
    """Фиксируем поведение pivot_table: без фильтра он усредняет — но громко.

    Логи идут через loguru, поэтому ни caplog (только stdlib), ни capfd (sink
    привязан к stderr до подмены) их не видят — подключаем свой sink."""
    from loguru import logger

    captured: list[str] = []
    sink = logger.add(lambda m: captured.append(str(m)), level="ERROR")
    try:
        pivot = make_pivot_table(_runs())
    finally:
        logger.remove(sink)

    assert pivot.loc["col1", "xgboost"] == round((0.835469 + 0.741553) / 2, 4)
    joined = "".join(captured)
    assert "УСРЕДНЕНИЕ" in joined and "col1/xgboost" in joined, "коллизия обязана быть видна"
    assert "data=" in joined, "в сообщении должно быть видно, чем различаются прогоны"


def test_filter_is_applied_in_fetch(monkeypatch):
    """fetch_benchmark_table с data_name обязан отбросить чужой домен."""
    import deepfakes.eval.report as rep

    runs = _runs()

    class Exp:
        experiment_id = "1"

    monkeypatch.setattr(rep.mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(rep.mlflow, "get_experiment_by_name", lambda *a, **k: Exp())
    monkeypatch.setattr(rep.mlflow, "search_runs", lambda *a, **k: runs.copy())

    out = rep.fetch_benchmark_table(experiment_name="deepfake_source_bias", data_name="gan_only")
    assert set(out["tags.data"]) == {"gan_only"}
    assert len(out) == 2

    out_dm = rep.fetch_benchmark_table(experiment_name="deepfake_source_bias", data_name="dm_only")
    assert len(out_dm) == 1


def test_missing_domain_returns_empty_not_wrong_table(monkeypatch):
    import deepfakes.eval.report as rep

    class Exp:
        experiment_id = "1"

    monkeypatch.setattr(rep.mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(rep.mlflow, "get_experiment_by_name", lambda *a, **k: Exp())
    monkeypatch.setattr(rep.mlflow, "search_runs", lambda *a, **k: _runs())

    out = rep.fetch_benchmark_table(experiment_name="deepfake_source_bias", data_name="combined")
    assert out.empty, "для отсутствующего домена лучше пустая таблица, чем чужие числа"


# --- 28.09.2026: строгость фильтра и схема валидации ---------------------------


def test_missing_tag_column_yields_nothing_not_everything():
    """Главная регрессия: без колонки тега фильтр раньше молча пропускался.

    У прогонов негативных контролей тега data не было, и отчёт по dm_only взял
    контроли GAN-домена, написав над ними «Контроли пройдены»."""
    from deepfakes.eval.report import filter_by_tag

    runs = pd.DataFrame(
        {
            "tags.mlflow.runName": ["negctrl__col2__xgboost"],
            "metrics.negctrl_trivial_filesize_auc": [0.497],
        }
    )
    out = filter_by_tag(runs, "data", "dm_only", "deepfake_source_bias_neg_ctrl")
    assert out.empty, "нет тега — нельзя утверждать принадлежность к домену"


def test_untagged_runs_are_dropped_tagged_kept():
    from deepfakes.eval.report import filter_by_tag

    runs = pd.DataFrame(
        {
            "tags.mlflow.runName": ["a", "b", "c"],
            "tags.data": ["dm_only", None, "gan_only"],
        }
    )
    out = filter_by_tag(runs, "data", "dm_only", "x")
    assert list(out["tags.mlflow.runName"]) == ["a"]


def test_no_value_means_no_filtering():
    """Фильтр без значения — осознанный «все прогоны», не молчаливый сбой."""
    from deepfakes.eval.report import filter_by_tag

    runs = pd.DataFrame({"tags.data": ["a", "b"]})
    assert len(filter_by_tag(runs, "data", None, "x")) == 2


def test_cv_filter_separates_holdout_from_kfold(monkeypatch):
    """Регрессия: пересборка gan_only во время k-fold усреднила holdout и k5."""
    import deepfakes.eval.report as rep

    runs = pd.DataFrame(
        {
            "tags.pipeline": ["col1", "col1"],
            "tags.model": ["logreg", "logreg"],
            "tags.data": ["gan_only", "gan_only"],
            "tags.cv": ["holdout", "k5_stratified"],
            "tags.seed": ["42", "42"],
            "metrics.test_roc_auc": [0.614965, 0.615451],
            "status": ["FINISHED", "FINISHED"],
        }
    )

    class Exp:
        experiment_id = "1"

    monkeypatch.setattr(rep.mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(rep.mlflow, "get_experiment_by_name", lambda *a, **k: Exp())
    monkeypatch.setattr(rep.mlflow, "search_runs", lambda *a, **k: runs.copy())

    out = rep.fetch_benchmark_table("deepfake_source_bias", data_name="gan_only", cv_name="holdout")
    assert list(out["tags.cv"]) == ["holdout"]
    assert find_collapsed_cells(out) == []
    assert make_pivot_table(out).loc["col1", "logreg"] == 0.615  # holdout, не среднее


def test_empty_controls_section_is_loud():
    """Отсутствие контролей — не нейтральный факт, а предупреждение."""
    from deepfakes.eval.report import _negative_control_section

    text = "".join(_negative_control_section(pd.DataFrame()))
    assert "НЕ ЗАПУСКАЛИСЬ" in text
    assert "пройдены" not in text.lower()


def test_filesize_only_signal_is_explained_as_content_floor():
    """Вес файла при одинаковой геометрии — планка, а не утечка подготовки."""
    from deepfakes.eval.report import _negative_control_section

    nc = pd.DataFrame(
        {
            "tags.mlflow.runName": ["negctrl__col2__xgboost"],
            "metrics.negctrl_trivial_geometry_auc": [0.5],
            "metrics.negctrl_trivial_luminance_auc": [0.52],
            "metrics.negctrl_trivial_filesize_auc": [0.71],
            "metrics.negctrl_resolution_only_auc": [0.72],
            "metrics.negctrl_shuffle_labels_auc": [0.5],
        }
    )
    text = "".join(_negative_control_section(nc))
    assert "ВНИМАНИЕ" in text
    assert "сжимаемости содержимого" in text
    assert "нижняя планка" in text.lower() or "нижней планк" in text.lower()


def test_splits_version_separates_leaky_from_clean(monkeypatch):
    """Прогон на v2 (утечка личностей) не должен подменять отсутствующий v3."""
    import deepfakes.eval.report as rep

    runs = pd.DataFrame(
        {
            "tags.pipeline": ["ela", "ela", "tex2"],
            "tags.model": ["xgboost"] * 3,
            "tags.data": ["dm_only"] * 3,
            "tags.cv": ["holdout"] * 3,
            "tags.splits_version": ["v2", "v3", "v2"],  # tex2 на v3 «упал»
            "tags.seed": ["42"] * 3,
            "metrics.test_roc_auc": [0.9946, 0.91, 0.9916],
            "status": ["FINISHED"] * 3,
        }
    )

    class Exp:
        experiment_id = "1"

    monkeypatch.setattr(rep.mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(rep.mlflow, "get_experiment_by_name", lambda *a, **k: Exp())
    monkeypatch.setattr(rep.mlflow, "search_runs", lambda *a, **k: runs.copy())

    out = rep.fetch_benchmark_table(
        "deepfake_source_bias", data_name="dm_only", cv_name="holdout", splits_version="v3"
    )
    assert list(out["tags.pipeline"]) == ["ela"], "v2-значение tex2 не должно просочиться"
    assert float(out["metrics.test_roc_auc"].iloc[0]) == 0.91
