"""Парный набор: общая предобработка, маска, сплит по исходнику (scripts/build_pairs.py)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts import build_pairs as bp


def _jpeg(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path, quality=90)


def test_split_of_is_stable_and_roughly_70_15_15():
    stems = [f"{i:05d}" for i in range(4000)]
    s = pd.Series([bp.split_of(x) for x in stems])
    assert [bp.split_of(x) for x in stems[:50]] == list(s[:50])
    frac = s.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03
    assert abs(frac["val"] - 0.15) < 0.03
    assert abs(frac["test"] - 0.15) < 0.03


def test_select_sources_is_deterministic_and_order_free():
    files = [Path(f"/x/{i:05d}.jpg") for i in range(100)]
    a = bp.select_sources(files, 10)
    b = bp.select_sources(list(reversed(files)), 10)
    assert a == b and len(a) == 10


def test_finish_is_256_jpeg_q95_without_subsampling():
    im = Image.fromarray(np.full((512, 512, 3), 120, np.uint8))
    out = Image.open(io.BytesIO(bp.finish(im)))
    assert out.size == (256, 256) and out.format == "JPEG"
    # 4:4:4 — у всех компонент одинаковые коэффициенты дискретизации
    assert len({(h, v) for _, h, v, _ in out.layer}) == 1
    # таблица яркости q95 из libjpeg: первый коэффициент 2
    assert out.quantization[0][0] == 2


def test_finish_rejects_wrong_size():
    with pytest.raises(AssertionError):
        bp.finish(Image.new("RGB", (256, 256)))


def test_face_mask_covers_center_not_corners():
    hard, soft = bp.face_mask(512)
    h = np.asarray(hard)
    assert h[int(0.56 * 512), 256] == 255
    assert h[5, 5] == 0 and h[-5, -5] == 0
    cover = (h > 0).mean()
    assert 0.25 < cover < 0.40  # эллипс π·0.29·0.36 ≈ 0.33 кадра
    # размытая маска мягкая только у шва
    s = np.asarray(soft)
    assert s[int(0.56 * 512), 256] == 255 and s[5, 5] == 0


def test_composite_keeps_background_bit_exact():
    orig = Image.fromarray(np.random.default_rng(0).integers(0, 255, (512, 512, 3), np.uint8))
    gen = Image.new("RGB", (512, 512), (0, 255, 0))
    _, soft = bp.face_mask(512)
    out = np.asarray(bp.composite(gen, orig, soft))
    bg = np.asarray(soft) == 0
    assert np.array_equal(out[bg], np.asarray(orig)[bg])
    assert (out[np.asarray(soft) == 255] == [0, 255, 0]).all()


def test_degenerate_detects_black():
    assert bp._is_degenerate(Image.new("RGB", (64, 64)))
    noise = np.random.default_rng(1).integers(0, 255, (64, 64, 3), np.uint8)
    assert not bp._is_degenerate(Image.fromarray(noise))


def test_real_and_stylegan_share_the_same_path(tmp_path, monkeypatch):
    """Без GPU: real и stylegan проходят один и тот же путь, запуск возобновляем."""
    src = tmp_path / "src"
    for i in range(6):
        _jpeg(src / "real" / f"{i:05d}.jpg", i)
        _jpeg(src / "fake" / f"F{i:04d}.jpg", 100 + i)
    out = tmp_path / "out"
    argv = ["x", "generate", "--src", str(src), "--out", str(out), "--n", "4",
            "--gens", "real", "stylegan"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    bp.main()
    reals = sorted((out / "real").glob("*.jpg"))
    fakes = sorted((out / "stylegan").glob("*.jpg"))
    assert len(reals) == 4 and len(fakes) == 4
    for p in reals + fakes:
        im = Image.open(p)
        assert im.size == (256, 256) and im.quantization[0][0] == 2
    # тот же исходник через тот же путь даёт побайтно тот же файл
    p0 = reals[0]
    expect = bp.finish(bp.up512(Image.open(src / "real" / p0.name)))
    assert p0.read_bytes() == expect
    before = {p: p.stat().st_mtime_ns for p in reals}
    bp.main()  # повтор: ничего не перезаписывается
    assert {p: p.stat().st_mtime_ns for p in reals} == before


def test_splits_keep_pairs_together_and_drop_incomplete(tmp_path):
    out = tmp_path / "out"
    stems = [f"{i:05d}" for i in range(40)]
    for g in ("real", "vae", "img2img", "inpaint"):
        for s in stems:
            if g == "inpaint" and s == "00003":
                continue  # неполная пара
            (out / g).mkdir(parents=True, exist_ok=True)
            (out / g / f"{s}.jpg").write_bytes(b"x")
    (out / "stylegan").mkdir()
    for i in range(10):
        (out / "stylegan" / f"S{i}.jpg").write_bytes(b"x")

    df = bp.build_rows(out)
    assert not df.path.str.contains("00003").any()
    assert (df.groupby("group").split.nunique() == 1).all()
    assert df.groupby("source").size().to_dict() == {
        "real_p": 39, "vae_p": 39, "img2img_p": 39, "inpaint_p": 39, "stylegan_p": 10
    }  # fmt: skip

    names = bp.write_splits(df, tmp_path / "splits", "v1")
    assert set(names) == {"pairs_vae", "pairs_img2img", "pairs_inpaint", "pairs_stylegan", "pairs_all"}
    # в каждом варианте один и тот же реальный кадр лежит в одном и том же сплите
    where = {}
    for n in names:
        for s in ("train", "val", "test"):
            part = pd.read_csv(tmp_path / "splits" / n / "v1" / f"{s}.csv")
            assert list(part.columns) == ["path", "label", "source", "group"]
            for p in part[part.label == 0].path:
                assert where.setdefault(p, s) == s
    vae_test = pd.read_csv(tmp_path / "splits" / "pairs_vae" / "v1" / "test.csv")
    assert set(vae_test.source) <= {"real_p", "vae_p"}
    assert vae_test.label.value_counts().nunique() == 1  # ровно пары 1:1


def test_source_history_matches_140k_encoding():
    im = Image.fromarray(np.random.default_rng(3).integers(0, 255, (256, 256, 3), np.uint8))
    out = bp.source_history(im)
    assert out.size == (256, 256) and out.mode == "RGB"
    buf = io.BytesIO()
    im.save(buf, format="JPEG", **bp.SRC_JPEG)
    enc = Image.open(io.BytesIO(buf.getvalue()))
    # 140k: q95 (яркость 2,1,1,2,2,4…) и 4:2:0
    assert list(enc.quantization[0])[:6] == [2, 1, 1, 2, 2, 4]
    assert len({(h, v) for _, h, v, _ in enc.layer}) == 2  # 4:2:0 — разные коэффициенты
    assert bp.as_source(Image.new("RGB", (512, 512), (9, 9, 9))).size == (256, 256)
