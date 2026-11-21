"""Запись кэша признаков должна быть атомарной (общий кэш у параллельных задач)."""

from __future__ import annotations

import numpy as np

from deepfakes.utils.caching import feature_cache_path, load_cached, save_cached


def test_save_cached_roundtrip_and_no_temp_left(tmp_path):
    p = feature_cache_path(tmp_path, "abc123")
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    paths = ["a.png", "b.png", "c.png", "d.png"]
    save_cached(p, X, paths=paths, valid_mask=np.array([1, 1, 0, 1], dtype=bool))

    assert p.exists() and p.suffix == ".npz"
    assert not list(tmp_path.glob(".*tmp*")), "временный файл не убран"

    got = load_cached(p, expected_paths=paths)
    assert got is not None
    np.testing.assert_array_equal(got[0], X)
    np.testing.assert_array_equal(got[1], np.array([True, True, False, True]))


def test_truncated_cache_is_a_miss_not_a_crash(tmp_path):
    """Обрезанный архив (задача убита посреди записи) должен читаться как промах."""
    p = feature_cache_path(tmp_path, "trunc")
    save_cached(p, np.zeros((3, 2), dtype=np.float32), paths=["a", "b", "c"])
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])
    assert load_cached(p, expected_paths=["a", "b", "c"]) is None


def test_overwrite_replaces_previous_content(tmp_path):
    p = feature_cache_path(tmp_path, "over")
    save_cached(p, np.ones((2, 2), dtype=np.float32), paths=["a", "b"])
    save_cached(p, np.full((2, 2), 7.0, dtype=np.float32), paths=["a", "b"])
    got = load_cached(p, expected_paths=["a", "b"])
    assert got is not None
    np.testing.assert_array_equal(got[0], np.full((2, 2), 7.0, dtype=np.float32))


def test_write_failure_is_warning_not_crash(tmp_path, monkeypatch):
    """The extracted matrix survives a disk quota error."""
    import numpy as np

    from deepfakes.utils import caching

    p = feature_cache_path(tmp_path, "nospace")

    def boom(*a, **k):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(caching.np, "savez_compressed", boom)
    save_cached(p, np.zeros((3, 2), dtype=np.float32), paths=["a", "b", "c"])  # не падает

    assert not p.exists(), "битого кэша остаться не должно"
    assert not list(tmp_path.glob(".*tmp*")), "временный файл обязан быть убран"
    assert load_cached(p, expected_paths=["a", "b", "c"]) is None  # честный промах


def test_non_oserror_still_propagates(tmp_path, monkeypatch):
    """Глушим только ошибки ввода-вывода: логическая ошибка должна быть видна."""
    import numpy as np
    import pytest

    from deepfakes.utils import caching

    def boom(*a, **k):
        raise ValueError("сломанный массив")

    monkeypatch.setattr(caching.np, "savez_compressed", boom)
    with pytest.raises(ValueError):
        save_cached(feature_cache_path(tmp_path, "bug"), np.zeros((2, 2)), paths=["a", "b"])
