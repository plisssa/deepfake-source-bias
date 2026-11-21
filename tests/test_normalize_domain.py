"""Нормализация домена: кроп без интерполяции, отбраковка неподходящих файлов."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scripts.normalize_domain import (
    SKIP_BAD,
    SKIP_GRAY,
    SKIP_MODE,
    SKIP_SMALL,
    normalize_dir,
    normalize_one,
)


def _write(path, size, mode="RGB", seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if mode == "RGB":
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, (size[1], size[0]), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB" if mode == "RGB" else "L").save(path)
    return path


def _jpeg_roundtrip(arr: np.ndarray, quality: int) -> np.ndarray:
    """Тот же энкодер и те же параметры, что в normalize_one.

    Сравнивать выход с исходными пикселями напрямую нельзя: JPEG не lossless.
    Зато если прогнать ОЖИДАЕМЫЙ кроп через то же кодирование, результат обязан
    совпасть бит в бит — это и доказывает, что кроп взят верно и, главное, что
    никакой интерполяции по пути не было."""
    import io

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, subsampling=0, optimize=False)
    buf.seek(0)
    return np.asarray(Image.open(buf))


def test_crop_does_not_interpolate(tmp_path):
    """Ресэмплинг гасит высокие частоты и создаёт асимметрию между классами —
    именно этот канал испортил прогон 30.07.2026. Здесь его быть не должно."""
    src = _write(tmp_path / "in.png", (512, 512), seed=1)
    dst = tmp_path / "out.jpg"
    assert normalize_one((str(src), str(dst), 256, 95, 0.0)) == "ok"

    orig = np.asarray(Image.open(src))
    got = np.asarray(Image.open(dst))
    assert got.shape == (256, 256, 3)
    np.testing.assert_array_equal(got, _jpeg_roundtrip(orig[128:384, 128:384], 95))


def test_center_crop_is_centered(tmp_path):
    src = _write(tmp_path / "in.png", (440, 660), seed=2)
    dst = tmp_path / "out.jpg"
    normalize_one((str(src), str(dst), 256, 95, 0.0))
    orig = np.asarray(Image.open(src))
    expected = orig[
        (660 - 256) // 2 : (660 - 256) // 2 + 256, (440 - 256) // 2 : (440 - 256) // 2 + 256
    ]
    np.testing.assert_array_equal(np.asarray(Image.open(dst)), _jpeg_roundtrip(expected, 95))


@pytest.mark.parametrize(
    "size,mode,reason",
    [
        ((200, 300), "RGB", SKIP_SMALL),  # одна сторона меньше целевой
        ((300, 300), "L", SKIP_MODE),  # градации серого не конвертируем
    ],
)
def test_skips(tmp_path, size, mode, reason):
    src = _write(tmp_path / "in.png", size, mode=mode)
    assert normalize_one((str(src), str(tmp_path / "out.jpg"), 256, 90, 0.0)) == reason
    assert not (tmp_path / "out.jpg").exists()


def test_unreadable_is_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert normalize_one((str(bad), str(tmp_path / "out.jpg"), 256, 90, 0.0)) == SKIP_BAD


def test_no_temp_files_left_and_idempotent(tmp_path):
    src = _write(tmp_path / "in.png", (300, 300), seed=3)
    dst = tmp_path / "out.jpg"
    normalize_one((str(src), str(dst), 256, 90, 0.0))
    first = dst.read_bytes()
    assert normalize_one((str(src), str(dst), 256, 90, 0.0)) == "ok"  # повтор не переписывает
    assert dst.read_bytes() == first
    assert not list(tmp_path.glob(".*tmp*"))


def test_normalize_dir_counts_and_makes_classes_uniform(tmp_path):
    """Разнородные источники на выходе должны стать однородными по метаданным."""
    a, b = tmp_path / "a", tmp_path / "b"
    for i in range(6):
        _write(a / f"{i}.png", (512, 512), seed=10 + i)
    for i in range(6):
        _write(b / f"sh/{i}.png", (440, 660), seed=20 + i)
    _write(b / "gray.png", (440, 660), mode="L")
    _write(b / "small.png", (100, 100))

    ra = normalize_dir(a, tmp_path / "na", 256, 90, jobs=2, limit=0)
    rb = normalize_dir(b, tmp_path / "nb", 256, 90, jobs=2, limit=0)
    assert ra["counts"]["ok"] == 6
    assert rb["counts"]["ok"] == 6
    assert rb["counts"][SKIP_MODE] == 1
    assert rb["counts"][SKIP_SMALL] == 1

    def meta(d):
        out = set()
        for p in sorted(d.glob("*.jpg")):
            with Image.open(p) as im:
                q = im.quantization
                out.add((im.size, im.mode, tuple(sorted(tuple(v) for v in q.values()))))
        return out

    ma, mb = meta(tmp_path / "na"), meta(tmp_path / "nb")
    assert {m[0] for m in ma} == {(256, 256)}
    assert {m[0] for m in mb} == {(256, 256)}
    assert {m[1] for m in ma} == {m[1] for m in mb} == {"RGB"}
    # главное: таблицы квантования совпадают → ELA/DCT больше не читают домен
    assert {m[2] for m in ma} == {m[2] for m in mb}


def test_limit_is_deterministic(tmp_path):
    src = tmp_path / "s"
    for i in range(10):
        _write(src / f"{i:02d}.png", (300, 300), seed=i)
    r1 = normalize_dir(src, tmp_path / "o1", 256, 90, jobs=2, limit=4)
    r2 = normalize_dir(src, tmp_path / "o2", 256, 90, jobs=2, limit=4)
    assert r1["taken"] == r2["taken"] == 4
    n1 = sorted(p.name for p in (tmp_path / "o1").glob("*.jpg"))
    n2 = sorted(p.name for p in (tmp_path / "o2").glob("*.jpg"))
    assert n1 == n2


def test_saturation_filter_is_symmetric(tmp_path):
    """Серый файл, сохранённый как RGB, должен отбраковываться наравне с JPEG/L.

    Именно этой симметрии не хватало: аудит 26.09.2026 нашёл 17.8% почти Ч/Б у
    фейков против 6.6% у реальных, потому что серые реальные выпадали как
    not_rgb, а их диффузионные пары проходили как формально RGB."""
    g = np.repeat(
        np.random.default_rng(5).integers(40, 200, (300, 300, 1), dtype=np.uint8), 3, axis=2
    )
    gray_rgb = tmp_path / "gray_rgb.png"
    Image.fromarray(g, mode="RGB").save(gray_rgb)
    color = _write(tmp_path / "color.png", (300, 300), seed=6)

    assert normalize_one((str(gray_rgb), str(tmp_path / "a.jpg"), 256, 90, 12.0)) == SKIP_GRAY
    assert normalize_one((str(color), str(tmp_path / "b.jpg"), 256, 90, 12.0)) == "ok"
    # без порога прежнее поведение сохраняется
    assert normalize_one((str(gray_rgb), str(tmp_path / "c.jpg"), 256, 90, 0.0)) == "ok"


def test_saturation_filter_equalizes_gray_fraction(tmp_path):
    """С фильтром доли почти Ч/Б у обоих классов должны стать нулевыми."""
    from scripts.audit_uniformity import gray_fraction

    a, b = tmp_path / "a", tmp_path / "b"
    rng = np.random.default_rng(7)
    for i in range(4):  # класс A: часть серых, сохранённых как RGB
        arr = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        if i < 2:
            g = rng.integers(40, 200, (300, 300, 1), dtype=np.uint8)
            arr = np.repeat(g, 3, axis=2)
        (a).mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="RGB").save(a / f"{i}.png")
    for i in range(4):  # класс B: все цветные
        _write(b / f"{i}.png", (300, 300), seed=50 + i)

    normalize_dir(a, tmp_path / "na", 256, 90, jobs=2, limit=0, min_sat=12.0)
    normalize_dir(b, tmp_path / "nb", 256, 90, jobs=2, limit=0, min_sat=12.0)
    fa = gray_fraction([str(p) for p in (tmp_path / "na").glob("*.jpg")])[0]
    fb = gray_fraction([str(p) for p in (tmp_path / "nb").glob("*.jpg")])[0]
    assert fa == 0.0 and fb == 0.0
