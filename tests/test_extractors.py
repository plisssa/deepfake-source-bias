"""Extractors: каждый выдаёт фиксированную форму и не падает на маленьком датасете."""

from __future__ import annotations

import numpy as np
import pytest

from deepfakes.features.color import ColorMomentsExtractor
from deepfakes.features.freq import HFEnergyExtractor
from deepfakes.features.grad import LoGExtractor, LoGFlattenExtractor
from deepfakes.features.texture import LBPExtractor


@pytest.fixture()
def few_paths(tiny_dataset):
    import glob

    real = sorted(glob.glob(f"{tiny_dataset['real_dir']}/*.png"))[:5]
    fake = sorted(glob.glob(f"{tiny_dataset['fake_dir']}/*.png"))[:5]
    return np.array(real + fake, dtype=object)


def test_color_extractor_shape(few_paths):
    e = ColorMomentsExtractor(img_size=64, patch_sizes=(16, 32), colorspace="RGB", verbose=False)
    X = e.transform(few_paths)
    assert X.shape[0] == len(few_paths)
    assert X.shape[1] == e.n_features_out_


def test_freq_extractor_shape(few_paths):
    e = HFEnergyExtractor(img_size=64, n_bins=6, verbose=False)
    X = e.transform(few_paths)
    assert X.shape[0] == len(few_paths)
    # n_bins + slope + flatness + mean + std = 10
    assert X.shape[1] == 6 + 1 + 1 + 2


def test_log_patch_extractor(few_paths):
    e = LoGExtractor(img_size=64, patch_size=16, verbose=False)
    X = e.transform(few_paths)
    assert X.shape[0] == len(few_paths)
    # 4 patches × (4 stats: density+std+kurt+skew) + 3 global = 19
    expected_min = (64 // 16) ** 2 * 4 + 3
    assert X.shape[1] == expected_min


def test_log_flatten_extractor(few_paths):
    e = LoGFlattenExtractor(img_size=32, verbose=False)
    X = e.transform(few_paths)
    assert X.shape[1] == 32 * 32


def test_lbp_uniform(few_paths):
    e = LBPExtractor(img_size=64, radii=(1, 2), method="uniform", verbose=False)
    X = e.transform(few_paths)
    # uniform: бинов = 8r + 2; конкатенация по r=1 (10) + r=2 (18) = 28
    assert X.shape[0] == len(few_paths)
    assert X.shape[1] == (8 * 1 + 2) + (8 * 2 + 2)


def test_lbp_simple_legacy(few_paths):
    e = LBPExtractor(img_size=64, radii=(1,), method="simple", verbose=False)
    X = e.transform(few_paths)
    # simple-roll → bins=256
    assert X.shape[1] == 256
