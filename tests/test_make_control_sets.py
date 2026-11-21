"""Контрольные наборы статьи: схема A без утечки личностей и контроль «только обработка»."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from scripts import make_control_sets as mcs


def _jpeg(path, size=(256, 256), seed=0):
    a = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path, quality=95)


def test_celeba_partitions_are_respected(tmp_path, monkeypatch):
    monkeypatch.setattr(mcs, "N_REAL", {"train": 5, "val": 3, "test": 3})
    ids = [f"{i:06d}.jpg" for i in range(1, 61)]
    part = pd.DataFrame({"image_id": ids, "partition": [0] * 40 + [1] * 10 + [2] * 10})
    part.to_csv(tmp_path / "p.csv", index=False)
    got = mcs.celeba_by_partition(tmp_path / "p.csv", tmp_path / "img")
    pid = dict(zip(part.image_id, part.partition))
    for split, k in mcs.PART.items():
        assert len(got[split]) == mcs.N_REAL[split]
        assert {pid[p.split("/")[-1]] for p in got[split]} == {k}
    assert got == mcs.celeba_by_partition(tmp_path / "p.csv", tmp_path / "img")  # детерминизм


def test_a2_keeps_fakes_and_their_split(tmp_path):
    src = tmp_path / "gan"; src.mkdir()
    for s, n in (("train", 4), ("val", 2), ("test", 2)):
        pd.DataFrame(
            {"path": [f"/r/{s}{i}.jpg" for i in range(n)] + [f"/f/{s}{i}.jpg" for i in range(n)],
             "label": [0] * n + [1] * n, "source": ["ffhq"] * n + ["sg"] * n,
             "group": [f"g{s}{i}" for i in range(2 * n)]}
        ).to_csv(src / f"{s}.csv", index=False)
    reals = {s: [f"/c/{s}{i}.jpg" for i in range(3)] for s in ("train", "val", "test")}
    mcs.write_a2(src, tmp_path / "out", reals)
    for s in ("train", "val", "test"):
        df = pd.read_csv(tmp_path / "out" / f"{s}.csv")
        old = pd.read_csv(src / f"{s}.csv")
        assert set(df[df.label == 1].path) == set(old[old.label == 1].path)
        assert set(df[df.label == 0].path) == set(reals[s])
        assert not df.path.str.startswith("/r/").any()  # реальные FFHQ заменены


def test_resample_groups_differ_only_in_geometry(tmp_path):
    src = tmp_path / "valid"
    for i in range(20):
        _jpeg(src / f"{i:05d}.jpg", seed=i)
    counts = mcs.build_resample(src, tmp_path / "out", 10, tmp_path / "splits")
    a = sorted((tmp_path / "out" / "as_real_A").glob("*.jpg"))
    b = sorted((tmp_path / "out" / "as_fake_A").glob("*.jpg"))
    assert len(a) == len(b) == 10
    assert not {p.name for p in a} & {p.name for p in b}  # разные исходники
    assert {Image.open(p).size for p in a} == {mcs.CELEBA_SIZE}
    assert {Image.open(p).size for p in b} == {(256, 256)}
    qa = {tuple(Image.open(p).quantization[0])[:4] for p in a}
    qb = {tuple(Image.open(p).quantization[0])[:4] for p in b}
    assert qa == qb  # один энкодер, одно качество
    df = pd.concat(pd.read_csv(tmp_path / "splits" / f"{s}.csv") for s in ("train", "val", "test"))
    assert len(df) == 20 and set(df.label) == {0, 1}
    assert sum(sum(v.values()) for v in counts.values()) == 20
