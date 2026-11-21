"""Splits: дисьюнктность train/val/test, стратификация, корректное сохранение/загрузка."""

from __future__ import annotations

from deepfakes.io.collect import DatasetBundle, DatasetSource
from deepfakes.io.splits import SplitConfig, load_split_csv, make_holdout_split, save_split_csv


def test_holdout_disjoint(tiny_dataset, tmp_path):
    bundle = DatasetBundle(
        real_sources=[
            DatasetSource(name="rt", path=tiny_dataset["real_dir"], label=0, use_subfolders=False)
        ],
        fake_sources=[
            DatasetSource(name="ft", path=tiny_dataset["fake_dir"], label=1, use_subfolders=False)
        ],
    )
    X, y, s, g = bundle.collect()
    splits = make_holdout_split(
        X, y, s, SplitConfig(test_size=0.25, val_size=0.25, seed=42), groups=g
    )

    a, b, c = set(splits["train"]["X"]), set(splits["val"]["X"]), set(splits["test"]["X"])
    assert a.isdisjoint(b)
    assert a.isdisjoint(c)
    assert b.isdisjoint(c)
    assert len(a) + len(b) + len(c) == len(X)


def test_split_io_roundtrip(tiny_dataset, tmp_path):
    bundle = DatasetBundle(
        real_sources=[
            DatasetSource(name="rt", path=tiny_dataset["real_dir"], label=0, use_subfolders=False)
        ],
        fake_sources=[
            DatasetSource(name="ft", path=tiny_dataset["fake_dir"], label=1, use_subfolders=False)
        ],
    )
    X, y, s, g = bundle.collect()
    splits = make_holdout_split(X, y, s, SplitConfig(test_size=0.25, val_size=0.25), groups=g)
    save_split_csv(splits, tmp_path)
    loaded = load_split_csv(tmp_path)
    for name in ("train", "val", "test"):
        assert list(loaded[name]["X"]) == list(splits[name]["X"])
        assert list(loaded[name]["y"]) == list(splits[name]["y"])
