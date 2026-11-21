"""Загрузка .env проекта (пути к данным, MLFLOW_TRACKING_URI и т.д.).

Раньше .env.example предлагал «скопировать в .env», но ничто его не загружало —
на сервере молча срабатывали локальные фолбэк-пути. Теперь каждый CLI-скрипт
вызывает load_project_dotenv() ДО композиции Hydra-конфига (oc.env резолвится
лениво, но подстрахуемся ранним вызовом на импорте скрипта).
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def load_project_dotenv() -> bool:
    """Найти и загрузить .env: рядом с корнем проекта или в CWD.

    Уже выставленные переменные окружения НЕ перезаписываются.
    Возвращает True, если файл найден и загружен.
    """
    global _loaded
    if _loaded:
        return True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    candidates = [
        (
            Path(os.environ.get("PROJECT_ROOT", "")) / ".env"
            if os.environ.get("PROJECT_ROOT")
            else None
        ),
        Path(__file__).resolve().parents[2] / ".env",  # корень репозитория
        Path.cwd() / ".env",
    ]
    for cand in candidates:
        if cand is not None and cand.is_file():
            load_dotenv(cand, override=False)
            _loaded = True
            return True
    return False
