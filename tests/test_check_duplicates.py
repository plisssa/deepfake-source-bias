"""Проверка дубликатов между сплитами: ловит копии, не выдумывает их."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from scripts.check_duplicates import cross_split_pairs, fingerprint


def _img(seed):
    return np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)


def test_recompressed_copy_has_same_dhash_different_md5(tmp_path):
    a = tmp_path / "a.png"
    Image.fromarray(_img(1)).save(a)
    b = tmp_path / "b.jpg"  # та же картинка, пересохранённая в JPEG
    Image.fromarray(_img(1)).save(b, quality=90)
    ma, da = fingerprint(str(a))
    mb, db = fingerprint(str(b))
    assert ma != mb, "разные байты"
    assert da == db, "визуальная копия обязана совпасть по dHash"


def test_different_images_differ(tmp_path):
    fps = []
    for i in range(20):
        p = tmp_path / f"{i}.png"
        Image.fromarray(_img(100 + i)).save(p)
        fps.append(fingerprint(str(p))[1])
    assert len(set(fps)) == 20


def test_unreadable_file_is_not_a_crash(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"nope")
    md5, dh = fingerprint(str(p))
    assert md5 and dh is None


def test_cross_split_pairs_only_reports_cross_split():
    df = pd.DataFrame(
        {
            "path": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "split": ["train", "train", "train", "test"],
            "md5": ["x", "x", "y", "y"],  # x — копия внутри train, y — через сплиты
        }
    )
    out = cross_split_pairs(df, "md5")
    assert list(out["key"]) == ["y"]
    assert out["splits"].iloc[0] == "test,train"


def test_confirm_visual_separates_true_copy_from_hash_collision(tmp_path):
    """Совпадение dHash у разных картинок не должно считаться дубликатом."""
    from scripts.check_duplicates import confirm_visual

    rng = np.random.default_rng(0)
    base = np.tile(np.linspace(40, 200, 64, dtype=np.float32), (64, 1))
    a = np.clip(base + rng.normal(0, 3, (64, 64)), 0, 255).astype(np.uint8)
    other = np.clip(base + rng.normal(0, 25, (64, 64)), 0, 255).astype(np.uint8)  # тот же градиент
    pa, pb, pc = tmp_path / "a.png", tmp_path / "b.jpg", tmp_path / "c.png"
    Image.fromarray(a).save(pa)
    Image.fromarray(a).save(pb, quality=92)  # настоящая копия
    Image.fromarray(other).save(pc)  # другая картинка, похожая грубо
    df = pd.DataFrame(
        {
            "path": [str(pa), str(pb), str(pc)],
            "label": [0, 0, 1],
            "split": ["train", "test", "test"],
            "dhash": [1, 1, 2],
        }
    )
    got = confirm_visual(df, pd.DataFrame({"key": [1]}))
    assert bool(got["confirmed"].iloc[0]), "пересохранённая копия подтверждается"

    df2 = df.assign(dhash=[5, 9, 5])  # коллизия хэша у a и c
    got2 = confirm_visual(df2, pd.DataFrame({"key": [5]}))
    assert not bool(got2["confirmed"].iloc[0]), "коллизия хэша — не дубликат"
