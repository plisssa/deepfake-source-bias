#!/usr/bin/env python3
"""Привести источники к общему низкоуровневому виду ДО экстракции признаков.

Зачем. Аудит 26.09.2026 показал, что diffusion-домен нельзя считать как есть:
DeepFakeFace отрендерен в 512×512, а его же реальная половина (wiki) — исходные
кропы IMDB-WIKI переменного размера (440×660, 440×587, 440×293…), с другой
таблицей квантования, и среди них ~12% не RGB (JPEG/L и PNG/RGBA). Любой из этих
каналов отделяет классы сам по себе, без всякой детекции — это ровно та ловушка,
что дала AUC ≈ 1.0 в июле.

Как приводим — и почему именно так:

* **Центральный кроп, а НЕ ресайз.** Ресайз означает интерполяцию, а интерполяция
  гасит высокие частоты — то есть стирает спектральный след генератора, ради
  которого весь частотный блок и существует. Хуже того, если один класс
  ресэмплится, а другой проходит как no-op, возникает асимметрия ресэмплинга:
  контрольный эксперимент в июле дал по этому каналу AUC 0.94 на обычных фото
  вообще без дипфейков. Кроп не интерполирует ничего и сохраняет нативные
  артефакты обоих классов.
* **Цена кропа** — разное поле зрения: из 512×512 вырезается четверть площади, из
  440×293 — почти всё. Это разница в содержании кадра, а не низкоуровневый
  артефакт; её видно глазом и она честно фиксируется в манифесте. Если источник
  меньше целевого размера, файл отбраковывается, а не апскейлится.
* **Одинаковое перекодирование JPEG** одним энкодером с одним качеством —
  выравнивает таблицы квантования, иначе ELA, blockiness и DCT-признаки читают
  историю сжатия.
* **Не-RGB отбраковываем**, а не конвертируем: серое изображение, растянутое в три
  канала, остаётся серым, и цветовые пайплайны прочтут это как метку класса.

Запуск:
    python scripts/normalize_domain.py \\
        --src data/raw/dff_extracted/wiki      --out data/raw/norm/wiki \\
        --src data/raw/dff_extracted/text2img  --out data/raw/norm/text2img \\
        --size 256 --quality 90 --jobs 8

После прогона ОБЯЗАТЕЛЬНО проверить результат аудитом:
    python scripts/audit_uniformity.py --dir OUT_A --dir OUT_B --sample 3000
и убедиться, что код возврата 0.

Автономен: PIL + stdlib, без torch/cv2 и без импорта пакета проекта, поэтому
безопасен для login-узла.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# причины отбраковки — попадают в манифест по счётчикам
SKIP_SMALL = "too_small"
SKIP_MODE = "not_rgb"
SKIP_BAD = "unreadable"
SKIP_GRAY = "near_gray"


def mean_saturation(img: Image.Image) -> float:
    """Средняя насыщенность как max(R,G,B) - min(R,G,B), шкала 0..255."""
    import numpy as np

    a = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.int16)
    return float((a.max(axis=2) - a.min(axis=2)).mean())


def normalize_one(args: tuple[str, str, int, int, float]) -> str:
    """Один файл: центральный кроп size×size + перекодирование JPEG.

    Возвращает "ok" либо причину отбраковки. Запись атомарна (временный файл +
    rename): каталог может готовиться несколькими процессами сразу."""
    src, dst, size, quality, min_sat = args
    dst_path = Path(dst)
    if dst_path.exists():
        return "ok"
    try:
        with Image.open(src) as img:
            # Режим проверяем ДО кропа: серые и RGBA не конвертируем, а исключаем
            if img.mode != "RGB":
                return SKIP_MODE
            w, h = img.size
            if w < size or h < size:
                return SKIP_SMALL
            # Фильтр насыщенности — СИММЕТРИЧНЫЙ, по одному критерию для всех
            # источников. Без него получается асимметрия, найденная аудитом
            # 26.09.2026: серые реальные отбраковывались как не-RGB, а их
            # диффузионные пары — формально RGB — оставались, и доля почти Ч/Б
            # расходилась 17.8% против 6.6%. Цветовые пайплайны прочли бы это
            # как метку класса.
            if min_sat > 0 and mean_saturation(img) < min_sat:
                return SKIP_GRAY
            left = (w - size) // 2
            top = (h - size) // 2
            crop = img.crop((left, top, left + size, top + size))
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst_path.with_name(f".{dst_path.stem}.{os.getpid()}.tmp{dst_path.suffix}")
            # Один энкодер и одно q → одинаковые таблицы квантования у обоих
            # классов. subsampling=0 (4:4:4) — сознательно: прореживание цветности
            # 4:2:0 вдвое режет разрешение хромы, а часть следов генератора живёт
            # именно там (цветовые пайплайны COL-* и работают с хромой). Для
            # однородности достаточно, чтобы обработка была ОДИНАКОВОЙ; терять
            # сигнал ради размера файла смысла нет.
            crop.save(tmp, format="JPEG", quality=quality, subsampling=0, optimize=False)
            os.replace(tmp, dst_path)
        return "ok"
    except Exception:
        return SKIP_BAD


def collect(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())


def normalize_dir(
    src: Path, out: Path, size: int, quality: int, jobs: int, limit: int, min_sat: float = 0.0
) -> dict:
    files = collect(src)
    total_found = len(files)
    if limit and len(files) > limit:
        files = files[:limit]  # sorted() выше делает отбор детерминированным
    out.mkdir(parents=True, exist_ok=True)
    # плоское имя с индексом: шарды исходника не воспроизводим, зато исключаем
    # коллизии одинаковых basename из разных шардов
    tasks = [
        (str(p), str(out / f"{i:06d}_{p.stem}.jpg"), size, quality, min_sat)
        for i, p in enumerate(files)
    ]

    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for res in ex.map(normalize_one, tasks, chunksize=64):
            counts[res] = counts.get(res, 0) + 1

    print(f"\n--- {src.name} → {out}")
    print(f"  найдено файлов:     {total_found}" + (f" (взято {len(files)})" if limit else ""))
    print(f"  записано:           {counts.get('ok', 0)}")
    for reason in (SKIP_MODE, SKIP_SMALL, SKIP_GRAY, SKIP_BAD):
        if counts.get(reason):
            print(f"  отбраковано {reason:<11} {counts[reason]}")
    return {
        "src": str(src),
        "out": str(out),
        "found": total_found,
        "taken": len(files),
        "size": size,
        "quality": quality,
        "min_sat": min_sat,
        "counts": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", action="append", default=[], required=True, help="каталог-источник")
    ap.add_argument("--out", action="append", default=[], required=True, help="куда писать (парно)")
    ap.add_argument("--size", type=int, default=256, help="сторона центрального кропа")
    ap.add_argument("--quality", type=int, default=90, help="качество JPEG на выходе")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("DEEPFAKES_N_JOBS", "8")))
    ap.add_argument("--limit", type=int, default=0, help="максимум файлов на источник (0 = все)")
    ap.add_argument(
        "--min-sat",
        type=float,
        default=0.0,
        help="отбраковывать изображения со средней насыщенностью ниже порога "
        "(0..255, 0 = не фильтровать; для diffusion-домена нужно 12)",
    )
    ap.add_argument("--manifest", default="", help="куда сложить JSON-отчёт")
    args = ap.parse_args()

    if len(args.src) != len(args.out):
        sys.exit(f"--src и --out должны идти парами: {len(args.src)} против {len(args.out)}")

    report = []
    for s, o in zip(args.src, args.out):
        report.append(
            normalize_dir(
                Path(s),
                Path(o),
                args.size,
                args.quality,
                args.jobs,
                args.limit,
                args.min_sat,
            )
        )

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nМанифест: {args.manifest}")

    print("\nТеперь ОБЯЗАТЕЛЬНО проверьте однородность:")
    outs = " ".join(f"--dir {o}" for o in args.out)
    print(f"  python scripts/audit_uniformity.py {outs} --sample 3000")
    print("Код 0 — можно считать. Код 2 — остался канал домена, считать нельзя.")


if __name__ == "__main__":
    main()
