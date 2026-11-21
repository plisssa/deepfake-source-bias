"""Общие фикстуры: генерация маленького набора синтетических изображений."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> dict:
    """Создаёт маленький фейковый датасет (40 real + 40 fake) во временной папке.

    Real: чистый шум.
    Fake: тот же шум + слабый высокочастотный паттерн (имитация GAN-артефакта),
    чтобы экстракторы могли что-то выучить.
    """
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("tiny_ds")
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    n_per = 40
    for i in range(n_per):
        img = (rng.normal(128, 20, (64, 64, 3))).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(real_dir / f"r_{i}.png")
        # fake: + лёгкая шахматка (высокочастотный паттерн)
        chess = np.zeros((64, 64, 3), dtype=np.float32)
        chess[::2, ::2] = 12
        chess[1::2, 1::2] = 12
        img2 = (rng.normal(128, 20, (64, 64, 3)) + chess).clip(0, 255).astype(np.uint8)
        Image.fromarray(img2).save(fake_dir / f"f_{i}.png")

    return {
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "n": 2 * n_per,
    }


@pytest.fixture()
def tiny_splits(tiny_dataset) -> dict:
    """Готовый holdout split на синтетическом датасете."""
    from deepfakes.io.collect import DatasetBundle, DatasetSource
    from deepfakes.io.splits import SplitConfig, make_holdout_split

    bundle = DatasetBundle(
        real_sources=[
            DatasetSource(name="rt", path=tiny_dataset["real_dir"], label=0, use_subfolders=False)
        ],
        fake_sources=[
            DatasetSource(name="ft", path=tiny_dataset["fake_dir"], label=1, use_subfolders=False)
        ],
    )
    X, y, s, g = bundle.collect()
    return make_holdout_split(
        X, y, s, SplitConfig(test_size=0.25, val_size=0.25, seed=42), groups=g
    )
