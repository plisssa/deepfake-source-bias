"""Поштучные предсказания test: сохраняются выровненными с путями."""

from __future__ import annotations

import numpy as np

from scripts.run_pipeline import save_test_predictions


def test_predictions_aligned_with_paths_after_dropping_invalid(tmp_path):
    paths = np.array(["a.png", "b.png", "c.png", "d.png"], dtype=object)
    valid = np.array([True, False, True, True])  # b не читается и из метрик выпал
    y = np.array([0, 1, 1])
    scores = np.array([0.1, 0.8, 0.7])
    p = save_test_predictions(tmp_path, paths, valid, y, scores, threshold=0.5)
    d = np.load(p, allow_pickle=False)
    assert list(d["path"]) == ["a.png", "c.png", "d.png"], "путь должен идти со СВОИМ скором"
    np.testing.assert_array_equal(d["y"], y)
    np.testing.assert_allclose(d["score"], scores)
    assert float(d["threshold"]) == 0.5


def test_loads_without_pickle(tmp_path):
    """Пути — строки, а не object: файл читается без allow_pickle (безопасно)."""
    p = save_test_predictions(
        tmp_path, np.array(["x"], dtype=object), np.array([True]), [1], [0.9], 0.5
    )
    assert np.load(p, allow_pickle=False)["path"].dtype.kind == "U"
