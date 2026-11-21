"""Сбор и валидация путей к изображениям + извлечение групп (анти-утечка)."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# Хвост имени файла, похожий на счётчик кадра: "_0001", "-frame12", "frame07".
# Обязателен разделитель или ключевое слово перед цифрами — иначе голые цифры,
# являющиеся частью ID ('video1' vs 'video12'), склеивали бы разные видео.
_FRAME_SUFFIX_RE = re.compile(
    r"(?:[\s_\-\.]+(?:frame|f|img|im)?|(?:frame|img|im))[\s_\-\.]*\d+$",
    re.IGNORECASE,
)


def collect_images(
    path: str | Path,
    exts: tuple[str, ...] = DEFAULT_EXTS,
    recursive: bool = True,
    verify: bool = True,
) -> list[str]:
    """Собрать все валидные изображения по пути.

    Args:
        path: корневая директория.
        exts: расширения (регистронезависимо).
        recursive: рекурсивный обход подпапок.
        verify: проверять целостность через PIL.

    Returns:
        Отсортированный список валидных путей.
    """
    path = str(path)
    found: set[str] = set()
    # оба регистра + set: на case-insensitive ФС (macOS) не будет дублей,
    # на case-sensitive (Linux-сервер) не потеряются *.JPG
    for ext in {e for base in exts for e in (base.lower(), base.upper())}:
        pattern = os.path.join(path, "**", ext) if recursive else os.path.join(path, ext)
        found.update(glob.glob(pattern, recursive=recursive))
    files = sorted(found)

    if not verify:
        return files
    return verify_images(files)


def _verify_one(f: str) -> bool:
    try:
        with Image.open(f) as img:
            img.verify()
        return True
    except Exception:
        return False


def verify_images(files: list[str], n_threads: int = 16) -> list[str]:
    """PIL-верификация целостности, параллельно потоками (I/O-bound).

    Порядок файлов сохраняется."""
    if not files:
        return files
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        ok = list(pool.map(_verify_one, files, chunksize=64))
    valid = [f for f, good in zip(files, ok) if good]
    bad = len(files) - len(valid)
    if bad:
        for f, good in zip(files, ok):
            if not good:
                log.warning(f"[BAD] {f}")
        log.warning(f"Skipped {bad} corrupted file(s) of {len(files)}")
    return valid


def _stem_prefix_group(path: str) -> str:
    """Группа по префиксу имени: '<dir>/video12_frame003.jpg' → '<dir>/video12'.

    Если после отбрасывания хвостового счётчика ничего не осталось (файл вида
    '000123.jpg'), группа = сам файл (группировки нет)."""
    p = Path(path)
    stem = _FRAME_SUFFIX_RE.sub("", p.stem)
    if not stem:
        return path
    return str(p.parent / stem)


def extract_groups(files: list[str], mode: str = "auto", regex: str | None = None) -> list[str]:
    """Извлечь group_id для каждого файла.

    Modes:
        none        — группа = сам файл (группировки нет);
        parent      — группа = родительская директория (кадры видео в папках);
        stem_prefix — группа = имя файла без хвостового счётчика кадров;
        regex       — первая capture-группа `regex` по полному пути;
        auto        — stem_prefix + проверка вменяемости: если эвристика
                      склеивает подозрительно много файлов в одну группу
                      (или не находит групп вовсе) — откат к none.
    """
    if mode == "none":
        return list(files)
    if mode == "parent":
        return [str(Path(f).parent) for f in files]
    if mode == "regex":
        if not regex:
            raise ValueError("group_mode='regex' требует group_regex")
        rx = re.compile(regex)
        out = []
        for f in files:
            m = rx.search(f)
            out.append(m.group(1) if m else f)
        return out
    if mode in ("stem_prefix", "auto"):
        groups = [_stem_prefix_group(f) for f in files]
        if mode == "stem_prefix":
            return groups
        # auto: sanity-проверки эвристики. Признаки артефакта именования (а не
        # реальных видео): группы в основном синглтоны ИЛИ один префикс
        # поглотил большую долю файлов. Абсолютные пороги по числу групп не
        # используются — они ломались на маленьких источниках (8 видео × 50).
        uniq, counts = np.unique(groups, return_counts=True)
        n = len(files)
        if len(uniq) == n:
            return list(files)  # группировки не найдено — каждый файл сам по себе
        largest_frac = counts.max() / max(n, 1)
        median_size = float(np.median(counts))
        if median_size <= 1 or largest_frac > 0.5:
            log.warning(
                f"group auto-heuristic rejected: {len(uniq)} groups, median size "
                f"{median_size:.0f}, largest covers {largest_frac:.0%} — "
                f"falling back to per-file groups (set group_mode explicitly if needed)"
            )
            return list(files)
        log.info(
            f"  groups: {len(uniq)} (median size={int(np.median(counts))}, "
            f"max={int(counts.max())}) via stem_prefix heuristic"
        )
        return groups
    raise ValueError(f"Unknown group_mode: {mode}")


@dataclass
class DatasetSource:
    """Один источник изображений с меткой и тегом источника."""

    name: str
    path: str
    label: int  # 0 = real, 1 = fake
    max_samples: int | None = None
    use_subfolders: bool = True
    group_mode: str = "auto"  # none | parent | stem_prefix | regex | auto
    group_regex: str | None = None
    # Детерминированная дизъюнктная половина источника по хэшу имени файла:
    # 'even'/'odd'. Нужно, когда ОДИН физический источник (например celeba)
    # используется и в train-, и в test-бандле cross-протокола — половины
    # гарантированно не пересекаются.
    hash_split: str | None = None
    # По чему считать хэш для hash_split: 'file' (имя файла, прежнее поведение)
    # или 'group' (ключ группы, например ID человека). 'group' нужен, когда
    # у одной личности несколько файлов в разных источниках: иначе половины
    # не пересекаются по файлам, но пересекаются по людям.
    hash_split_by: str = "file"
    # Пространство имён групп. По умолчанию — имя источника: группы разных
    # источников никогда не склеиваются (защита от случайно совпавших ID в
    # несвязанных датасетах). Источники с ОБЩИМИ личностями обязаны задать одно
    # и то же пространство — иначе человек 123 в real и в fake окажется в двух
    # разных группах и может попасть по разные стороны сплита.
    # Найдено 28.09.2026: DeepFakeFace генерирует фейки ИЗ wiki-фотографий, и
    # 36% фейков в test принадлежали людям, чьи фейки уже были в train.
    group_namespace: str | None = None

    def collect(self, seed: int = 42) -> tuple[list[str], list[str], list[str]]:
        # Верификация — ПОСЛЕ сабсэмплинга: на источниках в сотни тысяч файлов
        # PIL-verify всего каталога занимал десятки минут (TIMEOUT на кластере),
        # хотя нужны только max_samples из них.
        files = collect_images(self.path, recursive=self.use_subfolders, verify=False)
        if self.hash_split is not None:
            if self.hash_split not in ("even", "odd"):
                raise ValueError(f"hash_split must be 'even'|'odd', got {self.hash_split}")
            if self.hash_split_by not in ("file", "group"):
                raise ValueError(f"hash_split_by must be 'file'|'group', got {self.hash_split_by}")
            want = 0 if self.hash_split == "even" else 1
            import hashlib as _hashlib

            if self.hash_split_by == "group":
                keys = extract_groups(files, mode=self.group_mode, regex=self.group_regex)
            else:
                keys = [Path(f).name for f in files]
            files = [
                f
                for f, k in zip(files, keys)
                if int(_hashlib.sha1(str(k).encode()).hexdigest(), 16) % 2 == want
            ]
            log.info(
                f"[{self.name}] hash_split={self.hash_split} by {self.hash_split_by}: "
                f"{len(files)} files"
            )
        n_total = len(files)
        if self.max_samples is not None and n_total > self.max_samples:
            # запас ~5% на битые файлы: после верификации обрежем до max_samples
            oversample = min(n_total, int(self.max_samples * 1.05) + 50)
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_total, size=oversample, replace=False)
            files = [files[i] for i in sorted(idx)]
        files = verify_images(files)
        if self.max_samples is not None and len(files) > self.max_samples:
            files = files[: self.max_samples]
        if self.max_samples is not None:
            log.info(f"[{self.name}] sampled {len(files)} / {n_total} files (seed={seed})")
        groups = extract_groups(files, mode=self.group_mode, regex=self.group_regex)
        # префикс пространства имён: по умолчанию источник (группы разных
        # источников не склеиваются), либо явно заданное общее пространство
        ns = self.group_namespace or self.name
        groups = [f"{ns}::{g}" for g in groups]
        sources = [self.name] * len(files)
        return files, sources, groups


@dataclass
class DatasetBundle:
    """Набор real- и fake-источников, который собирается в единый X/y/sources/groups."""

    real_sources: list[DatasetSource] = field(default_factory=list)
    fake_sources: list[DatasetSource] = field(default_factory=list)

    def collect(self, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_files: list[str] = []
        all_labels: list[int] = []
        all_sources: list[str] = []
        all_groups: list[str] = []

        for src in self.real_sources:
            files, srcs, grps = src.collect(seed=seed)
            all_files.extend(files)
            all_labels.extend([0] * len(files))
            all_sources.extend(srcs)
            all_groups.extend(grps)
            log.info(f"  REAL [{src.name}] -> {len(files)} files")

        for src in self.fake_sources:
            files, srcs, grps = src.collect(seed=seed)
            all_files.extend(files)
            all_labels.extend([1] * len(files))
            all_sources.extend(srcs)
            all_groups.extend(grps)
            log.info(f"  FAKE [{src.name}] -> {len(files)} files")

        if not all_files:
            raise ValueError("No images collected — check dataset paths.")

        X = np.array(all_files, dtype=object)
        y = np.array(all_labels, dtype=np.int8)
        s = np.array(all_sources, dtype=object)
        g = np.array(all_groups, dtype=object)
        log.info(
            f"Bundle collected: total={len(X)}, real={int((y == 0).sum())}, "
            f"fake={int((y == 1).sum())}"
        )
        return X, y, s, g
