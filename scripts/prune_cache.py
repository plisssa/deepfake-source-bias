"""Чистка кэша признаков: оставить только кэши текущих канонических сплитов.

Зачем. 29.09.2026 кэш вырос до 129 ГБ (684 npz + 11 ГБ деградированных копий)
при квоте scratch 210 ГБ на весь аккаунт — k-fold кэширует каждый фолд
отдельно, плюс варианты на 256, блоки LOGO, устаревшие сплиты v2. Кэш — это
только оптимизация: любой удалённый файл пересчитается при необходимости.

Каждый npz хранит список путей, из которых он посчитан (CACHE_VERSION ≥ 3).
Файл ОСТАВЛЯЕТСЯ, если его список путей совпадает с одним из канонических
сплитов (например gan_only/v1/test) — их переиспользуют статистика, стенды
устойчивости и проверки. Всё остальное — кэши фолдов, сплитов прежних версий,
блоков LOGO и деградаций — удаляется. Без --apply ничего не удаляет.

    python -m scripts.prune_cache --keep gan_only/v1 dm_only/v3           # план
    python -m scripts.prune_cache --keep gan_only/v1 dm_only/v3 --apply   # удалить
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from deepfakes.utils.env import load_project_dotenv

load_project_dotenv()

REPO = Path(__file__).resolve().parents[1]


def canonical_keys(splits_root: Path, keep: list[str]) -> dict[frozenset, str]:
    """Множество путей каждого канонического сплита → его имя."""
    out: dict[frozenset, str] = {}
    for k in keep:
        for s in ("train", "val", "test"):
            f = splits_root / k / f"{s}.csv"
            if f.exists():
                out[frozenset(pd.read_csv(f)["path"].astype(str))] = f"{k}/{s}"
    return out


def classify(npz: Path, canon: dict[frozenset, str]) -> str | None:
    """Имя канонического сплита, если кэш посчитан ровно по нему; иначе None."""
    try:
        with np.load(npz, allow_pickle=True) as d:
            if "paths" not in d.files:
                return None
            paths = frozenset(str(p) for p in d["paths"])
    except Exception:
        return None  # битый файл — тоже в удаление
    return canon.get(paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", nargs="+", required=True, help="сплиты вида gan_only/v1")
    ap.add_argument("--cache", default=str(REPO / "data/processed/cache"))
    ap.add_argument("--splits", default=str(REPO / "data/splits"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    canon = canonical_keys(Path(args.splits), args.keep)
    print(f"канонических сплитов: {len(canon)}")
    keep_n = keep_b = drop_n = drop_b = 0
    by_split: dict[str, int] = {}
    to_drop: list[Path] = []
    for f in sorted(cache.glob("feat_*.npz")):
        name = classify(f, canon)
        size = f.stat().st_size
        if name:
            keep_n += 1
            keep_b += size
            by_split[name] = by_split.get(name, 0) + 1
        else:
            drop_n += 1
            drop_b += size
            to_drop.append(f)
    extra_dirs = [d for d in (cache / "degraded", cache / "processing_control") if d.exists()]
    extra_b = sum(p.stat().st_size for d in extra_dirs for p in d.rglob("*") if p.is_file())
    gb = 1024**3
    print(f"оставить: {keep_n} файлов, {keep_b / gb:.1f} ГБ — {by_split}")
    print(f"удалить:  {drop_n} файлов, {drop_b / gb:.1f} ГБ (фолды, v2, LOGO, кросс-бандлы, битые)")
    print(f"каталоги-производные: {[d.name for d in extra_dirs]}, {extra_b / gb:.1f} ГБ")
    if not args.apply:
        print("сухой прогон — ничего не удалено; для удаления --apply")
        return
    for f in to_drop:
        f.unlink(missing_ok=True)
    for d in extra_dirs:
        if d.name == "degraded":  # processing_control нужен текущей проверке — не трогаем
            shutil.rmtree(d, ignore_errors=True)
    print(f"удалено: {drop_n} файлов + degraded")


if __name__ == "__main__":
    main()
