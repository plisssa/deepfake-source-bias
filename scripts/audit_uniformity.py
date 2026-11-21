#!/usr/bin/env python3
"""Проверка ФАКТИЧЕСКОЙ однородности классов по метаданным файлов.

Зачем. Прогон 30.07.2026 дал ROC-AUC ≈ 1.0, потому что real и fake пришли из
разных пайплайнов подготовки (178×218 против 256×256): апскейл с искажением
пропорций + разная история JPEG давали до AUC 0.94 вообще без дипфейков.
Вывод аудита: однородность источников надо ПРОВЕРЯТЬ, а не принимать на веру
по описанию датасета. Пять минут работы закрывают самый дорогой класс ошибок.

Что считает (по каждому классу отдельно, потом сравнивает):
  * распределение (width, height) и соотношения сторон;
  * формат/режим (JPEG/PNG, RGB/L);
  * размер файла (StyleGAN-изображения глаже и при том же качестве весят меньше);
  * ХЕШ ТАБЛИЦЫ КВАНТОВАНИЯ JPEG — прямой отпечаток «каким энкодером и с каким
    качеством сжато»; если у классов он разный, модель может читать сжатие,
    а не артефакты генерации.

Дополнительно с `--pixels N`: декодирует N изображений на класс и считает долю
практически чёрно-белых (насыщенность ниже порога). Это не придирка: после
нормализации diffusion-домена 26.09.2026 у реальной половины (wiki) отбраковалось
2924 не-RGB файла, а у фейков — ни одного, потому что диффузия отдаёт формально
RGB даже из серого входа. Метаданные такую асимметрию не видят, а цветовые
пайплайны прочли бы её как метку класса.

Автономен: только PIL + numpy + stdlib, без torch/cv2 и без импорта пакета
проекта. Без `--pixels` читает лишь заголовки и безопасен для login-узла;
с `--pixels` декодирует пиксели — это уже счёт, запускайте через sbatch.

Запуск:
    python scripts/audit_uniformity.py --split data/splits/gan_only/v1/train.csv
    python scripts/audit_uniformity.py --dir DIR_REAL --dir DIR_FAKE --sample 4000
    python scripts/audit_uniformity.py --dir A --dir B --sample 3000 --pixels 500
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def probe(path: str) -> dict | None:
    """Метаданные одного файла без декодирования пикселей."""
    try:
        p = Path(path)
        size = p.stat().st_size
        with Image.open(p) as img:
            w, h = img.size
            fmt = img.format or "?"
            mode = img.mode
            # таблицы квантования есть только у JPEG; хешируем их состав
            q = getattr(img, "quantization", None)
            if q:
                blob = json.dumps({k: list(v) for k, v in sorted(q.items())}).encode()
                qhash = hashlib.sha1(blob).hexdigest()[:10]
            else:
                qhash = "none"
        return {
            "w": w,
            "h": h,
            "wh": f"{w}x{h}",
            "aspect": round(w / h, 3) if h else 0.0,
            "fmt": fmt,
            "mode": mode,
            "bytes": size,
            "qhash": qhash,
        }
    except Exception as e:  # битый файл — тоже факт, о котором надо знать
        print(f"  [BAD] {path}: {e}", file=sys.stderr)
        return None


#: ниже этой средней насыщенности (0..255) картинку считаем практически серой
GRAY_SAT_THRESHOLD = 12.0


def gray_fraction(paths: list[str]) -> tuple[float, int]:
    """Доля почти чёрно-белых изображений. Возвращает (доля, сколько прочитано).

    Насыщенность берём как max(R,G,B) - min(R,G,B) по пикселю: это S из HSV с
    точностью до масштаба, но без перевода цветового пространства. Серый файл,
    сохранённый как RGB, метаданными неотличим от цветного — а здесь виден."""
    import numpy as np

    vals: list[float] = []
    for p in paths:
        try:
            with Image.open(p) as img:
                small = img.convert("RGB").resize((64, 64))
                a = np.asarray(small, dtype=np.int16)
                vals.append(float((a.max(axis=2) - a.min(axis=2)).mean()))
        except Exception:
            continue
    if not vals:
        return float("nan"), 0
    arr = np.asarray(vals)
    return float((arr < GRAY_SAT_THRESHOLD).mean()), len(vals)


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    sizes = Counter(r["wh"] for r in rows)
    fmts = Counter(f"{r['fmt']}/{r['mode']}" for r in rows)
    qs = Counter(r["qhash"] for r in rows)
    byts = sorted(r["bytes"] for r in rows)
    med = byts[n // 2] if n else 0
    print(f"\n--- класс {label}: {n} файлов")
    print(f"  размеры (топ-3):      {', '.join(f'{k}×{v}' for k, v in sizes.most_common(3))}")
    print(f"  формат/режим:         {', '.join(f'{k}×{v}' for k, v in fmts.most_common(3))}")
    print(f"  JPEG-квантование:     {', '.join(f'{k}×{v}' for k, v in qs.most_common(3))}")
    print(
        f"  размер файла, КБ:     медиана {med / 1024:.1f}, "
        f"p10 {byts[n // 10] / 1024:.1f}, p90 {byts[9 * n // 10] / 1024:.1f}"
    )
    return {"sizes": sizes, "fmts": fmts, "qs": qs, "median_bytes": med, "n": n}


def compare(a: dict, b: dict, name_a: str, name_b: str) -> tuple[list[str], list[str]]:
    """Возвращает (блокирующие проблемы, предупреждения).

    Разделение появилось 27.09.2026. Аудит нормализованного diffusion-домена
    блокировал счёт из-за разницы веса файла в 1.51×, хотя размер, формат и
    таблицы квантования у классов совпадали. Но если история сжатия одинаковая,
    разница веса не может идти ОТ сжатия — она идёт от содержимого: диффузионные
    изображения глаже и потому сжимаются лучше. Это и есть искомый признак
    генератора, а не артефакт подготовки, и блокировать его нельзя. Порог 1.5×
    при этом резал ровно посередине эффекта: пара wiki×inpainting проходила
    (1.46×), а wiki×text2img и wiki×insight — нет (1.51× и 1.52×).
    """
    problems: list[str] = []
    notes: list[str] = []

    def top_set(c: Counter, frac: float = 0.05) -> set:
        total = sum(c.values()) or 1
        return {k for k, v in c.items() if v / total >= frac}

    encoding_differs = False
    if top_set(a["sizes"]).isdisjoint(top_set(b["sizes"])):
        encoding_differs = True
        problems.append(
            f"РАЗМЕРЫ не пересекаются: {name_a}={list(top_set(a['sizes']))[:3]} "
            f"против {name_b}={list(top_set(b['sizes']))[:3]} → один класс будет "
            f"ресэмплиться, другой нет (канал домена, переживающий resize)"
        )
    if top_set(a["fmts"]).isdisjoint(top_set(b["fmts"])):
        encoding_differs = True
        problems.append(
            f"ФОРМАТ/РЕЖИМ различаются: {top_set(a['fmts'])} против {top_set(b['fmts'])}"
        )
    if top_set(a["qs"]).isdisjoint(top_set(b["qs"])):
        encoding_differs = True
        problems.append(
            f"JPEG-КВАНТОВАНИЕ не пересекается: разная история сжатия "
            f"({name_a}: {list(top_set(a['qs']))[:2]}, {name_b}: {list(top_set(b['qs']))[:2]}) "
            f"→ питает ELA/blockiness/DCT-признаки независимо от геометрии"
        )
    ma, mb = a["median_bytes"], b["median_bytes"]
    if ma and mb:
        ratio = max(ma, mb) / min(ma, mb)
        if ratio >= 1.5 and encoding_differs:
            problems.append(
                f"РАЗМЕР ФАЙЛА различается в {ratio:.2f}× по медиане "
                f"({ma / 1024:.1f} КБ против {mb / 1024:.1f} КБ) при РАЗНОЙ истории "
                f"сжатия — тривиальный признак класса"
            )
        elif ratio >= 1.15:
            notes.append(
                f"вес файла различается в {ratio:.2f}× по медиане "
                f"({ma / 1024:.1f} КБ против {mb / 1024:.1f} КБ), но размер, формат и "
                f"квантование совпадают → разница от СОДЕРЖИМОГО (сгенерированные "
                f"изображения глаже и сжимаются лучше). Считать можно, но величину "
                f"обязательно привести в отчёте рядом с метриками и оценить негативным "
                f"контролем группы filesize"
            )
    ga, gb = a.get("gray_frac"), b.get("gray_frac")
    known = ga is not None and gb is not None and not (math.isnan(ga) or math.isnan(gb))
    # 5 процентных пунктов: столько даёт отбраковка серых у одного класса
    if known and abs(ga - gb) >= 0.05:
        problems.append(
            f"ДОЛЯ ПОЧТИ Ч/Б изображений различается: {name_a} {ga:.1%} против "
            f"{name_b} {gb:.1%} → насыщенность цвета становится признаком класса "
            f"(цветовые пайплайны COL-* прочтут её вместо артефактов генерации)"
        )
    return problems, notes


def collect_from_split(path: Path) -> dict[str, list[str]]:
    by_label: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_label.setdefault(str(row["label"]), []).append(row["path"])
    return by_label


def collect_from_dirs(dirs: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in dirs:
        files = [str(p) for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
        out[Path(d).name or d] = files
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", help="CSV сплита (колонки path,label)")
    ap.add_argument("--dir", action="append", default=[], help="каталог класса (можно дважды)")
    ap.add_argument("--sample", type=int, default=3000, help="файлов на класс (default 3000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pixels",
        type=int,
        default=0,
        help="декодировать N изображений на класс для проверки насыщенности "
        "(0 = не декодировать; с этим флагом запускайте через sbatch, не на login)",
    )
    args = ap.parse_args()

    if args.split:
        groups = collect_from_split(Path(args.split))
    elif len(args.dir) >= 2:
        groups = collect_from_dirs(args.dir)
    else:
        sys.exit("укажите --split CSV или как минимум два --dir")

    rng = random.Random(args.seed)
    stats: dict[str, dict] = {}
    for label, files in sorted(groups.items()):
        pick = files if len(files) <= args.sample else rng.sample(files, args.sample)
        rows = [r for r in (probe(p) for p in pick) if r is not None]
        if not rows:
            sys.exit(f"класс {label}: не удалось прочитать ни один файл")
        stats[label] = summarize(rows, label)
        if args.pixels:
            sub = pick if len(pick) <= args.pixels else rng.sample(pick, args.pixels)
            frac, n_read = gray_fraction(sub)
            stats[label]["gray_frac"] = frac
            print(f"  почти Ч/Б:            {frac:.1%} (по {n_read} декодированным)")

    labels = sorted(stats)
    print("\n" + "=" * 72)
    if len(labels) != 2:
        print(f"классов {len(labels)}, парное сравнение пропущено")
        return
    problems, notes = compare(stats[labels[0]], stats[labels[1]], labels[0], labels[1])
    if problems:
        print("РЕЗУЛЬТАТ: классы НЕ однородны — метрики будут завышены\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        if notes:
            print("\nКроме того (не блокирует):")
            for n in notes:
                print(f"  • {n}")
        print("\nЛечение: брать классы из одного пайплайна подготовки либо привести все")
        print("источники к общему разрешению и качеству JPEG ДО экстракции признаков.")
        sys.exit(2)
    print("РЕЗУЛЬТАТ: по метаданным подготовка классов одинакова (размеры, формат, сжатие)")
    if notes:
        print("\nВНИМАНИЕ — различия от содержимого, считать можно, но их надо измерить:")
        for n in notes:
            print(f"  • {n}")
    print("\nЭто необходимое, но не достаточное условие: семантика съёмки и выравнивание")
    print("лица метаданными не проверяются — их закрывает кросс-источниковый контроль.")


if __name__ == "__main__":
    main()
