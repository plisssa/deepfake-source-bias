"""Чистка кэша: канонические кэши остаются, остальное — в удаление."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.prune_cache import canonical_keys, classify


def test_keeps_only_exact_canonical_path_lists(tmp_path):
    sp = tmp_path / "splits" / "gan_only" / "v1"
    sp.mkdir(parents=True)
    pd.DataFrame({"path": ["a", "b", "c"], "label": [0, 1, 0]}).to_csv(sp / "test.csv", index=False)
    canon = canonical_keys(tmp_path / "splits", ["gan_only/v1"])

    def npz(name, paths):
        f = tmp_path / f"feat_{name}.npz"
        np.savez(f, X=np.zeros((len(paths), 2)), paths=np.array(paths, dtype=object))
        return f

    assert classify(npz("same", ["c", "a", "b"]), canon) == "gan_only/v1/test"  # порядок не важен
    assert classify(npz("fold", ["a", "b"]), canon) is None  # подмножество — фолд
    old = tmp_path / "feat_old.npz"
    np.savez(old, X=np.zeros((1, 1)))  # старый формат без путей
    assert classify(old, canon) is None
    bad = tmp_path / "feat_bad.npz"
    bad.write_bytes(b"nope")
    assert classify(bad, canon) is None
