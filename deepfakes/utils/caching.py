"""Кеширование признаков. Ключ — хэш по списку путей + параметрам extractor-а.

Гарантии корректности:
  * пути хэшируются В ИСХОДНОМ ПОРЯДКЕ (перестановка того же множества файлов
    даёт другой ключ — признаки не могут разъехаться с метками);
  * список путей сохраняется внутри .npz и сверяется при загрузке; если на
    диске лежит тот же набор в другом порядке — признаки переиндексируются;
  * в ключ входит CACHE_VERSION — при изменении формата кэша старые файлы
    молча игнорируются.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

# Поднимать при несовместимых изменениях формата кэша или логики экстракции.
# v3: HybridExtractor читает файл один раз (grayscale через cvtColor — признаки
# гибридов могут отличаться на округление от старых).
# v4: мат.фиксы экстракторов (азимутальный slope, flatness без DC, окно Ханна,
# нормировка кольцевых бинов, HSV range 0..180 + circular Hue, порядок YCbCr,
# LoG-порог/сигма, Canny L2, center_crop для blockiness/PRNU) — признаки
# freq/color/grad/noise изменились.
CACHE_VERSION = 4


def hash_inputs(paths: list[str], params: dict[str, Any]) -> str:
    """Хэшировать список файлов (имя+mtime+size, в исходном порядке) и параметры."""
    h = hashlib.sha256()
    h.update(f"cache_v{CACHE_VERSION}".encode())
    for p in paths:
        h.update(str(p).encode())
        try:
            st = os.stat(p)
            h.update(str(int(st.st_mtime)).encode())
            h.update(str(st.st_size).encode())
        except OSError:
            pass
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def feature_cache_path(cache_dir: str | Path, key: str) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"feat_{key}.npz"


def load_cached(
    cache_path: Path, expected_paths: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Загрузить кэш: (X, valid_mask). Если переданы expected_paths — сверить и
    при необходимости переиндексировать под их порядок. Кэш без сохранённых
    путей (старый формат) считается невалидным."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        valid = data["valid"].astype(bool) if "valid" in data else np.ones(X.shape[0], dtype=bool)
        if expected_paths is not None:
            if "paths" not in data:
                log.warning(
                    f"Cache {cache_path.name} has no stored paths — ignoring (stale format)"
                )
                return None
            stored = [str(p) for p in data["paths"].tolist()]
            expected = [str(p) for p in expected_paths]
            if stored != expected:
                if sorted(stored) != sorted(expected):
                    log.warning(f"Cache {cache_path.name}: path set mismatch — ignoring")
                    return None
                index = {p: i for i, p in enumerate(stored)}
                order = [index[p] for p in expected]
                X = X[order]
                valid = valid[order]
                log.info(f"Cache {cache_path.name}: reordered to match requested path order")
            if X.shape[0] != len(expected):
                return None
        log.info(f"Loaded cached features: {cache_path.name}")
        return X, valid
    except Exception as e:
        log.warning(f"Cache load failed for {cache_path}: {e}")
        return None


def save_cached(
    cache_path: Path,
    X: np.ndarray,
    paths: list[str] | None = None,
    valid_mask: np.ndarray | None = None,
) -> None:
    arrays: dict[str, np.ndarray] = {"X": X}
    if paths is not None:
        arrays["paths"] = np.array([str(p) for p in paths], dtype=object)
    if valid_mask is not None:
        arrays["valid"] = np.asarray(valid_mask, dtype=bool)
    # Пишем через временный файл и rename: кэш общий для параллельных задач
    # (несколько sbatch делят data/processed/cache), а запись npz «на месте»
    # оставляет читателю шанс подхватить обрезанный архив. load_cached такой
    # файл переживёт — но заплатит повторной экстракцией, а это часы.
    # Расширение .npz обязано остаться последним: savez сам дописывает его,
    # если не нашёл.
    tmp = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, cache_path)
    except OSError as e:
        # A failed cache write must not discard extracted features.
        log.warning(
            f"Не удалось записать кэш {cache_path.name}: {e}. "
            f"Прогон продолжается БЕЗ кэша (повторная экстракция обойдётся дороже). "
            f"Проверьте доступное место и квоту файловой системы."
        )
        return
    finally:
        if tmp.exists():
            tmp.unlink()
    log.info(f"Cached features: {cache_path.name} shape={X.shape}")
