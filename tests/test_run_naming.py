"""Имена прогонов и папок результатов — по ВАРИАНТУ конфига, не по классу.

Одна и та же ошибка всплыла трижды (28–29.09.2026): разные конфиги делят класс
пайплайна (rob_deep_fusion/rob1_fusion, tex5/tex5_r256, clip/dinov2 → deep_lp),
а имя собиралось из `cfg.pipeline.name` (класс). Итог — подмена в отчёте и
перезапись локальных результатов (tex5 на 128 затёрт tex5_r256). Тест находит
такие места статически, чтобы ошибка не вернулась в новом скрипте."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BAD = re.compile(r"(run_name|out_dir|reports/runs).*\{cfg\.pipeline\.name\}")


def test_no_run_name_built_from_pipeline_class():
    offenders = []
    for f in sorted(SCRIPTS.glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if BAD.search(line):
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, "имя по классу, а не по варианту:\n" + "\n".join(offenders)
