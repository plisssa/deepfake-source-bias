"""Проверка насыщенности в аудите: серый файл, сохранённый как RGB, должен ловиться."""

from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.audit_uniformity import compare, gray_fraction


def _save(path, gray: bool, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if gray:
        g = rng.integers(40, 200, (64, 64), dtype=np.uint8)
        arr = np.stack([g, g, g], axis=2)  # серое, но формально RGB
    else:
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return str(path)


def test_gray_fraction_detects_rgb_stored_grayscale(tmp_path):
    grays = [_save(tmp_path / f"g{i}.png", True, i) for i in range(6)]
    colors = [_save(tmp_path / f"c{i}.png", False, 100 + i) for i in range(6)]
    assert gray_fraction(grays)[0] == 1.0
    assert gray_fraction(colors)[0] == 0.0
    mixed = gray_fraction(grays[:3] + colors[:3])
    assert mixed[0] == 0.5 and mixed[1] == 6


def test_gray_fraction_survives_unreadable(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    ok = _save(tmp_path / "ok.png", True, 1)
    frac, n = gray_fraction([str(bad), ok])
    assert n == 1 and frac == 1.0


def test_gray_fraction_empty_is_nan():
    frac, n = gray_fraction([])
    import math

    assert n == 0 and math.isnan(frac)


def _stats(gray_frac):
    """Минимальная сводка: остальные каналы специально совпадают."""
    from collections import Counter

    return {
        "sizes": Counter({"256x256": 100}),
        "fmts": Counter({"JPEG/RGB": 100}),
        "qs": Counter({"aaa": 100}),
        "median_bytes": 20000,
        "n": 100,
        "gray_frac": gray_frac,
    }


def test_compare_flags_gray_asymmetry():
    problems, _ = compare(_stats(0.12), _stats(0.002), "wiki", "text2img")
    assert len(problems) == 1
    assert "Ч/Б" in problems[0]


def test_compare_allows_small_gray_difference():
    assert compare(_stats(0.03), _stats(0.01), "a", "b")[0] == []


def test_compare_skips_gray_check_without_pixels():
    """Без --pixels поля нет — проверка не должна выдумывать проблему."""
    a, b = _stats(0.12), _stats(0.0)
    del a["gray_frac"], b["gray_frac"]
    assert compare(a, b, "a", "b")[0] == []


def test_filesize_gap_is_a_note_when_encoding_matches():
    """Одинаковая подготовка + разный вес = признак содержимого, не блокер.

    Именно этот случай 27.09.2026 блокировал нормализованный diffusion-домен:
    1.51× по весу при совпадающих размере, формате и таблицах квантования."""
    a, b = _stats(0.01), _stats(0.01)
    a["median_bytes"], b["median_bytes"] = 23200, 34900
    problems, notes = compare(a, b, "text2img", "wiki")
    assert problems == [], "разница веса при одинаковой подготовке не должна блокировать"
    assert len(notes) == 1 and "СОДЕРЖИМОГО" in notes[0]


def test_filesize_gap_still_blocks_when_encoding_differs():
    """А вот при разной истории сжатия вес — настоящий домен-лик (июль 2026)."""
    from collections import Counter

    a, b = _stats(0.01), _stats(0.01)
    a["sizes"], b["sizes"] = Counter({"178x218": 100}), Counter({"256x256": 100})
    a["qs"], b["qs"] = Counter({"d4ba5ebcc1": 100}), Counter({"753360f6e7": 100})
    a["median_bytes"], b["median_bytes"] = 6700, 35500
    problems, _ = compare(a, b, "celeba", "text2img")
    assert any("РАЗМЕР ФАЙЛА" in p for p in problems)
    assert any("РАЗМЕРЫ" in p for p in problems)
