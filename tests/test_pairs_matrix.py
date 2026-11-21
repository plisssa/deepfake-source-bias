"""Полы и парная метрика матрицы переноса на парном наборе."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image

from scripts import run_pairs_matrix as rpm


def _save(tmp_path, name, arr, quality=None):
    p = tmp_path / name
    im = Image.fromarray(arr)
    if quality:
        im.save(p, format="JPEG", quality=quality)
    else:
        im.save(p, format="PNG")
    return str(p)


def test_blockiness_sees_jpeg_grid(tmp_path):
    rng = np.random.default_rng(0)
    smooth = (np.add.outer(np.arange(256), np.arange(256)) / 2).astype(np.uint8)
    img = np.clip(smooth[..., None] + rng.normal(0, 6, (256, 256, 3)), 0, 255).astype(np.uint8)
    png = _save(tmp_path, "a.png", img)
    jpg = _save(tmp_path, "b.jpg", img, quality=20)
    assert rpm.blockiness(jpg) > rpm.blockiness(png)
    assert abs(rpm.blockiness(png) - 1.0) < 0.1


def test_directed_floor_uses_train_direction():
    y = np.array([0, 0, 1, 1])
    f_tr = np.array([3.0, 4.0, 1.0, 2.0])  # на train фейки МЕНЬШЕ
    f_te = np.array([1.0, 2.0, 3.0, 4.0])  # на test наоборот
    assert rpm.directed_floor(f_tr, y, f_te, y) == 0.0  # не подглядываем в test


def test_pair_win_compares_within_pair():
    df = pd.DataFrame(
        {
            "group": ["a", "b", "c", "a", "b", "c"],
            "label": [0, 0, 0, 1, 1, 1],
            "score": [0.1, 0.9, 0.5, 0.2, 0.3, 0.6],
        }
    )
    # a: 0.2>0.1 да, b: 0.3>0.9 нет, c: 0.6>0.5 да
    assert rpm.pair_win(df) == 2 / 3


def test_pair_win_nan_without_pairs():
    df = pd.DataFrame({"group": ["a", "b"], "label": [0, 1], "score": [0.1, 0.2]})
    assert np.isnan(rpm.pair_win(df))
