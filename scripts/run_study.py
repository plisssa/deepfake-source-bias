"""Run the selected benchmark stages and record progress for restart."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger

load_project_dotenv()
log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "reports" / "study"
JOURNAL = STUDY_DIR / "journal.json"

# Пайплайны для cross-dataset / rob9 / negctrl — представители каждой группы
# (полный перебор всех 18 в этих протоколах умножил бы время впустую;
# состав можно править здесь).
CROSS_PIPELINES = ["col2", "freq4", "tex2", "grad4", "noise1", "rob1_fusion"]
NEGCTRL_PIPELINES = ["col2", "freq4", "tex2"]
# стенд устойчивости: представители блоков, по которым литература предсказывает
# разное поведение при пост-обработке (частоты рушатся, текстуры держатся)
ROBUST_PIPELINES = ["freq1", "freq4", "tex2", "tex5", "col2", "ela", "noise1"]
# Optuna: те, у кого есть пространство поиска в PIPELINE_SPACES
TUNE_PIPELINES = ["freq4", "col2", "tex2", "grad4", "noise1"]


def _py(module: str, *overrides: str) -> list[str]:
    return [sys.executable, "-m", module, *overrides]


def _subset(default: list[str], requested: str | None) -> list[str]:
    """Сузить состав стадии до указанных пайплайнов.

    Нужно, чтобы длинную стадию (robustness, tune) резать на несколько sbatch и
    считать параллельно: один job на пайплайн укладывается в лимит времени, а
    весь стенд целиком — нет. Журнал ключуется полной командой, поэтому части не
    затирают друг друга и resume работает по каждой отдельно."""
    if not requested:
        return default
    wanted = [p.strip() for p in requested.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in default]
    if unknown:
        sys.exit(f"Не из состава стадии: {unknown}. Доступны: {default}")
    return wanted


def build_stages(args) -> dict[str, list[list[str]]]:
    """Стадия → список команд. Порядок словаря = порядок выполнения."""
    common = [f"seed={args.seed}", f"tracking={args.tracking}", f"data={args.data}"]
    cv = f"cv={args.cv}"
    stages: dict[str, list[list[str]]] = {}

    stages["download"] = [
        [
            sys.executable,
            "-m",
            "scripts.download_data",
            *(["--only", args.download_only] if args.download_only else []),
        ]
    ]
    stages["splits"] = [_py("scripts.prepare_splits", cv, *common)]
    stages["benchmark"] = [_py("scripts.run_benchmark", cv, *common)]
    stages["cross_dataset"] = [
        _py(
            "scripts.run_cross_dataset",
            f"pipeline={p}",
            "model=xgboost",
            f"data={args.cross_data}",
            f"seed={args.seed}",
            f"tracking={args.tracking}",
        )
        for p in CROSS_PIPELINES
    ]
    stages["rob9"] = [
        _py(
            "scripts.run_rob9_cross_training",
            f"pipeline={p}",
            "model=xgboost",
            f"data={args.cross_data}",
            f"seed={args.seed}",
            f"tracking={args.tracking}",
        )
        for p in CROSS_PIPELINES
    ]
    stages["fusion"] = [_py("scripts.run_score_fusion", "experiment=rob2_score_fusion", *common)]
    stages["negative_control"] = [
        _py("scripts.run_negative_control", f"pipeline={p}", "model=xgboost", *common)
        for p in NEGCTRL_PIPELINES
    ]
    stages["interpret"] = [
        _py("scripts.run_interpret", f"pipeline={p}", "+interpret.kind=both", *common)
        for p in NEGCTRL_PIPELINES
    ]
    stages["robustness"] = [
        _py("scripts.run_robustness", f"pipeline={p}", "model=xgboost", *common)
        for p in _subset(ROBUST_PIPELINES, args.robust_pipelines)
    ]
    stages["tune"] = [
        _py("scripts.tune_pipeline", f"pipeline={p}", "model=xgboost", "tuning=optuna_50", *common)
        for p in _subset(TUNE_PIPELINES, args.tune_pipelines)
    ]
    # cv обязателен: отчёт фильтрует по схеме валидации, и без аргумента
    # k-fold-исследование молча получило бы holdout-таблицу (конфиг по умолчанию)
    stages["report"] = [_py("scripts.build_report", cv, *common)]
    return stages


def load_journal() -> dict:
    if JOURNAL.exists():
        try:
            return json.loads(JOURNAL.read_text())
        except json.JSONDecodeError as e:
            log.warning(f"Журнал повреждён ({e}); начинаю с чистого — стадии перепрогонятся")
    return {}


def _merge_journal(disk: dict, mine: dict) -> dict:
    """Влить свои записи в то, что уже на диске, не затирая чужие стадии."""
    out = dict(disk)
    for stage, entry in mine.items():
        if stage not in out:
            out[stage] = entry
            continue
        merged = dict(out[stage])
        merged.update({k: v for k, v in entry.items() if k != "commands"})
        cmds = dict(merged.get("commands", {}))
        cmds.update(entry.get("commands", {}))
        merged["commands"] = cmds
        out[stage] = merged
    return out


def save_journal(journal: dict) -> None:
    """Атомарная запись со слиянием: журнал делят параллельные sbatch.

    Стадии robustness и tune режутся на несколько задач (`--robust-pipelines`),
    и все они пишут этот файл. Без слияния тот, кто сохранился позже, затёр бы
    записи остальных, и при перезапуске чужие выполненные команды прогнались бы
    заново. Без rename читатель мог бы увидеть обрезанный JSON."""
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    # Сливаем ТОЛЬКО для файла и не трогаем переданный словарь: run_stage держит
    # ссылку на вложенную запись стадии (`entry`), и если подменить содержимое
    # journal, эта ссылка станет указывать на осиротевшую копию — все следующие
    # команды стадии запишутся в никуда. Именно так 26.09.2026 у cross_dataset
    # в журнал попали 2 команды из 6 и пропал статус стадии.
    merged = _merge_journal(load_journal(), journal)
    tmp = JOURNAL.with_name(f".{JOURNAL.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    os.replace(tmp, JOURNAL)


def run_stage(name: str, commands: list[list[str]], journal: dict) -> bool:
    """Выполнить стадию; в журнал — статус каждой команды. True = успех."""
    entry = journal.setdefault(name, {"commands": {}})
    entry["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    ok = True
    for cmd in commands:
        key = " ".join(cmd[2:])  # без интерпретатора
        prev = entry["commands"].get(key, {})
        if prev.get("status") == "ok":
            log.info(f"  [skip] {key} (уже выполнено)")
            continue
        log.info(f"  [run ] {key}")
        rc = subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
        entry["commands"][key] = {
            "status": "ok" if rc == 0 else "failed",
            "returncode": rc,
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        save_journal(journal)
        if rc != 0:
            log.error(f"  [FAIL] {key} (rc={rc})")
            ok = False
    entry["status"] = "ok" if ok else "failed"
    entry["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_journal(journal)
    return ok


def main() -> None:
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", help="через запятую; по умолчанию все по порядку")
    ap.add_argument("--force", help="стадии, которые перепрогнать даже если ok")
    ap.add_argument("--dry-run", action="store_true", help="показать план и выйти")
    ap.add_argument("--cv", default="k5_stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="default")
    ap.add_argument(
        "--cross-data",
        default="cross_gan_dm",
        help="data-конфиг для стадий cross_dataset/rob9 (train/test бандлы)",
    )
    ap.add_argument("--tracking", default="mlflow_local")
    ap.add_argument(
        "--robust-pipelines",
        help=f"подмножество стадии robustness через запятую (по умолчанию все: {ROBUST_PIPELINES})",
    )
    ap.add_argument(
        "--tune-pipelines",
        help=f"подмножество стадии tune через запятую (по умолчанию все: {TUNE_PIPELINES})",
    )
    ap.add_argument("--download-only", help="--only для download_data")
    ap.add_argument(
        "--skip-download", action="store_true", help="данные уже на месте — не скачивать"
    )
    args = ap.parse_args()

    stages = build_stages(args)
    if args.skip_download:
        stages.pop("download", None)
    selected = list(stages)
    if args.stages:
        wanted = [s.strip() for s in args.stages.split(",")]
        unknown = set(wanted) - set(stages)
        if unknown:
            sys.exit(f"Unknown stages: {sorted(unknown)}. Available: {list(stages)}")
        selected = [s for s in stages if s in wanted]

    if args.dry_run:
        print("План исследования:")
        for s in selected:
            print(f"\n[{s}]")
            for cmd in stages[s]:
                print("  " + " ".join(cmd[2:]))
        return

    journal = load_journal()
    force = {s.strip() for s in (args.force or "").split(",") if s.strip()}
    t0 = datetime.datetime.now()
    failed_stages: list[str] = []

    for s in selected:
        if s in force and s in journal:
            journal[s] = {"commands": {}}
        # Пропускаем стадию только если выполнены ИМЕННО ТЕ команды, что в плане
        # сейчас. Раньше проверялся лишь статус стадии — и `--stages benchmark
        # --cv k5_stratified` молча не делал ничего, потому что стадия была
        # закрыта прогоном с cv=holdout.
        done = journal.get(s, {}).get("commands", {})
        planned = [" ".join(cmd[2:]) for cmd in stages[s]]
        if planned and all(done.get(k, {}).get("status") == "ok" for k in planned):
            log.info(f"=== [{s}] все {len(planned)} команд уже выполнены, пропуск ===")
            continue
        log.info(f"=== [{s}] {len(stages[s])} команд ===")
        if not run_stage(s, stages[s], journal):
            failed_stages.append(s)
            log.error(
                f"=== [{s}] есть падения — цепочка остановлена; "
                f"после исправления перезапустите: продолжится с этого места ==="
            )
            break

    dt = datetime.datetime.now() - t0
    if failed_stages:
        log.error(f"Study FAILED на стадии {failed_stages[0]} (прошло {dt}). Журнал: {JOURNAL}")
        sys.exit(1)
    log.info(f"Study завершено за {dt}. Журнал: {JOURNAL}. Итоги: reports/tables/benchmark.*")


if __name__ == "__main__":
    main()
