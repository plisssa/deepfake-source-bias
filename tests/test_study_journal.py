"""Журнал должен выдерживать двух писателей одновременно."""
from __future__ import annotations

import json


def test_merge_keeps_both_writers(tmp_path, monkeypatch):
    import scripts.run_study as rs

    monkeypatch.setattr(rs, "STUDY_DIR", tmp_path)
    monkeypatch.setattr(rs, "JOURNAL", tmp_path / "journal.json")

    # задача A завершила свою команду
    a = {"robustness": {"commands": {"cmd_freq4": {"status": "ok"}}, "status": "ok"}}
    rs.save_journal(a)
    # задача B читала журнал ДО записи A (её снимок пуст) и сохраняет своё
    b = {"robustness": {"commands": {"cmd_tex2": {"status": "ok"}}, "status": "ok"}}
    rs.save_journal(b)

    on_disk = json.loads((tmp_path / "journal.json").read_text())
    cmds = on_disk["robustness"]["commands"]
    assert "cmd_freq4" in cmds, "запись задачи A затёрта"
    assert "cmd_tex2" in cmds, "запись задачи B потеряна"
    assert not list(tmp_path.glob(".*tmp*")), "временный файл не убран"


def test_other_stages_survive(tmp_path, monkeypatch):
    import scripts.run_study as rs

    monkeypatch.setattr(rs, "STUDY_DIR", tmp_path)
    monkeypatch.setattr(rs, "JOURNAL", tmp_path / "journal.json")
    rs.save_journal({"benchmark": {"commands": {"c1": {"status": "ok"}}, "status": "ok"}})
    rs.save_journal({"tune": {"commands": {"c2": {"status": "ok"}}, "status": "ok"}})
    on_disk = json.loads((tmp_path / "journal.json").read_text())
    assert set(on_disk) == {"benchmark", "tune"}


def test_corrupt_journal_is_not_fatal(tmp_path, monkeypatch):
    import scripts.run_study as rs

    monkeypatch.setattr(rs, "STUDY_DIR", tmp_path)
    monkeypatch.setattr(rs, "JOURNAL", tmp_path / "journal.json")
    (tmp_path / "journal.json").write_text('{"benchmark": {"comm')
    assert rs.load_journal() == {}
    rs.save_journal({"tune": {"commands": {}, "status": "ok"}})
    assert "tune" in json.loads((tmp_path / "journal.json").read_text())


def test_save_does_not_orphan_stage_entry(tmp_path, monkeypatch):
    """run_stage держит ссылку на запись стадии — save_journal не должен её подменять.

    Регрессия 26.09.2026: слияние заменяло содержимое journal, ссылка `entry` в
    run_stage начинала указывать на осиротевшую копию, и все команды стадии после
    первой в журнал не попадали (у cross_dataset осталось 2 записи из 6)."""
    import json

    import scripts.run_study as rs

    monkeypatch.setattr(rs, "STUDY_DIR", tmp_path)
    monkeypatch.setattr(rs, "JOURNAL", tmp_path / "journal.json")

    journal: dict = {}
    entry = journal.setdefault("cross_dataset", {"commands": {}})
    for i in range(6):
        entry["commands"][f"cmd{i}"] = {"status": "ok"}
        rs.save_journal(journal)
    entry["status"] = "ok"
    rs.save_journal(journal)

    on_disk = json.loads((tmp_path / "journal.json").read_text())
    assert on_disk["cross_dataset"]["status"] == "ok"
    assert len(on_disk["cross_dataset"]["commands"]) == 6, "часть команд потеряна"


def test_stage_skipped_only_when_planned_commands_done(tmp_path, monkeypatch, caplog):
    """Смена cv меняет состав команд — стадию пропускать нельзя."""
    import scripts.run_study as rs

    monkeypatch.setattr(rs, "STUDY_DIR", tmp_path)
    monkeypatch.setattr(rs, "JOURNAL", tmp_path / "journal.json")

    class A:
        seed, tracking, data, cv = 42, "mlflow_local", "gan_only", "holdout"
        cross_data, download_only = "cross_gan_dm", None
        robust_pipelines = tune_pipelines = None

    holdout = rs.build_stages(A())
    A.cv = "k5_stratified"
    kfold = rs.build_stages(A())

    key_holdout = " ".join(holdout["benchmark"][0][2:])
    key_kfold = " ".join(kfold["benchmark"][0][2:])
    assert key_holdout != key_kfold, "cv обязан входить в ключ команды"

    done = {"benchmark": {"status": "ok", "commands": {key_holdout: {"status": "ok"}}}}
    planned = [" ".join(c[2:]) for c in kfold["benchmark"]]
    assert not all(
        done["benchmark"]["commands"].get(k, {}).get("status") == "ok" for k in planned
    ), "k-fold прогон был бы пропущен как уже выполненный"
