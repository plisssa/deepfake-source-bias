"""Тесты имён признаков: get_feature_names() согласован с фактической размерностью.

Главный инвариант: после transform длина списка имён строго равна X.shape[1]
для каждого основного экстрактора; плюс спот-чеки осмысленных имён и
префиксация в Hybrid/Fusion.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from deepfakes.features.color import ColorHistogramExtractor, ColorMomentsExtractor
from deepfakes.features.forensic import ELAExtractor
from deepfakes.features.freq import (
    DCTBlockExtractor,
    FullFFTExtractor,
    HFEnergyExtractor,
    WaveletExtractor,
)
from deepfakes.features.fusion import FeatureFusionExtractor
from deepfakes.features.grad import (
    CannyEdgeExtractor,
    LoGExtractor,
    ScharrExtractor,
    SobelExtractor,
)
from deepfakes.features.hybrid import HybridExtractor
from deepfakes.features.noise import (
    JPEGBlockinessExtractor,
    NoiseResidualSRMExtractor,
    PRNUExtractor,
)
from deepfakes.features.texture import (
    GaborBankExtractor,
    GLCMExtractor,
    HOGExtractor,
    LBPExtractor,
    LPQExtractor,
    LTPExtractor,
)


@pytest.fixture(scope="module")
def image_paths(tmp_path_factory) -> np.ndarray:
    """3 синтетические RGB-картинки на диске (пути — вход экстракторов)."""
    rng = np.random.default_rng(42)
    d = tmp_path_factory.mktemp("feature_names_imgs")
    paths = []
    for i in range(3):
        img = rng.integers(0, 256, size=(80, 96, 3), dtype=np.uint8)
        p = d / f"img_{i}.png"
        Image.fromarray(img).save(p)
        paths.append(str(p))
    return np.array(paths)


# Все основные экстракторы (маленький img_size — ради скорости).
EXTRACTOR_FACTORIES = [
    pytest.param(lambda: HFEnergyExtractor(img_size=64, verbose=False), id="HFEnergy"),
    pytest.param(lambda: FullFFTExtractor(img_size=64, verbose=False), id="FullFFT"),
    pytest.param(lambda: DCTBlockExtractor(img_size=64, verbose=False), id="DCTBlock"),
    pytest.param(lambda: WaveletExtractor(img_size=64, verbose=False), id="Wavelet"),
    pytest.param(
        lambda: ColorMomentsExtractor(img_size=64, patch_sizes=(32,), verbose=False),
        id="ColorMoments",
    ),
    pytest.param(
        lambda: ColorMomentsExtractor(
            img_size=64, patch_sizes=(32,), colorspace="HSV", full_stats=False, verbose=False
        ),
        id="ColorMoments-hsv-short",
    ),
    pytest.param(lambda: ColorHistogramExtractor(img_size=64, verbose=False), id="ColorHistogram"),
    pytest.param(
        lambda: ColorHistogramExtractor(img_size=64, patch_grid=2, verbose=False),
        id="ColorHistogram-grid",
    ),
    pytest.param(lambda: LBPExtractor(img_size=64, verbose=False), id="LBP"),
    pytest.param(lambda: HOGExtractor(img_size=64, verbose=False), id="HOG"),
    pytest.param(lambda: GLCMExtractor(img_size=64, verbose=False), id="GLCM"),
    pytest.param(lambda: GLCMExtractor(img_size=64, patch_grid=2, verbose=False), id="GLCM-grid"),
    pytest.param(lambda: GaborBankExtractor(img_size=64, verbose=False), id="Gabor"),
    pytest.param(
        lambda: GaborBankExtractor(img_size=64, patch_grid=2, verbose=False), id="Gabor-grid"
    ),
    pytest.param(lambda: LTPExtractor(img_size=64, verbose=False), id="LTP"),
    pytest.param(lambda: LPQExtractor(img_size=64, verbose=False), id="LPQ"),
    pytest.param(lambda: SobelExtractor(img_size=64, verbose=False), id="Sobel"),
    pytest.param(lambda: ScharrExtractor(img_size=64, verbose=False), id="Scharr"),
    pytest.param(lambda: CannyEdgeExtractor(img_size=64, verbose=False), id="Canny"),
    pytest.param(lambda: LoGExtractor(img_size=64, verbose=False), id="LoG"),
    pytest.param(
        lambda: NoiseResidualSRMExtractor(img_size=64, verbose=False), id="NoiseResidualSRM"
    ),
    pytest.param(
        lambda: NoiseResidualSRMExtractor(img_size=64, patch_grid=2, verbose=False),
        id="NoiseResidualSRM-grid",
    ),
    pytest.param(lambda: JPEGBlockinessExtractor(img_size=64, verbose=False), id="JPEGBlockiness"),
    pytest.param(lambda: PRNUExtractor(img_size=64, verbose=False), id="PRNU"),
    pytest.param(lambda: ELAExtractor(img_size=64, verbose=False), id="ELA"),
]


@pytest.mark.parametrize("factory", EXTRACTOR_FACTORIES)
def test_names_match_output_dimension(factory, image_paths):
    """Главный тест: имена обязаны совпадать с фактической размерностью выхода."""
    ext = factory()
    X = ext.transform(image_paths)
    names = ext.get_feature_names()
    assert names is not None, f"{type(ext).__name__}: get_feature_names() вернул None"
    assert (
        len(names) == X.shape[1]
    ), f"{type(ext).__name__}: имён {len(names)}, а признаков {X.shape[1]}"
    assert len(set(names)) == len(names), f"{type(ext).__name__}: имена не уникальны"
    assert all(isinstance(n, str) and n for n in names)


def test_hf_energy_spot_names(image_paths):
    ext = HFEnergyExtractor(img_size=64, verbose=False)
    ext.transform(image_paths)
    names = ext.get_feature_names()
    assert names[0] == "ring_00"
    assert "ring_07" in names
    assert names[8:] == ["slope", "flatness", "mean", "std"]


def test_gabor_spot_names(image_paths):
    ext = GaborBankExtractor(img_size=64, verbose=False)
    ext.transform(image_paths)
    names = ext.get_feature_names()
    assert names[0] == "theta0_lam4_mean"
    assert "theta135_lam16_energy" in names


def test_lbp_spot_names(image_paths):
    ext = LBPExtractor(img_size=64, verbose=False)
    ext.transform(image_paths)
    names = ext.get_feature_names()
    assert names[0] == "r1_bin00"
    assert any(n.startswith("r3_bin") for n in names)


def test_color_moments_spot_names(image_paths):
    ext = ColorMomentsExtractor(img_size=64, patch_sizes=(32,), verbose=False)
    ext.transform(image_paths)
    names = ext.get_feature_names()
    assert names[0] == "patch32_y0_x0_chR_mean"
    assert "patch32_y32_x32_chB_iqr" in names


def test_srm_and_ela_spot_names(image_paths):
    srm = NoiseResidualSRMExtractor(img_size=64, verbose=False)
    srm.transform(image_paths)
    srm_names = srm.get_feature_names()
    assert srm_names[0] == "residual_mean"
    assert "srm5_abs_mean" in srm_names

    ela = ELAExtractor(img_size=64, verbose=False)
    ela.transform(image_paths)
    ela_names = ela.get_feature_names()
    assert ela_names[0] == "global_R_mean"
    assert "g3_3_B_std" in ela_names


def test_hybrid_prefixed_names(image_paths):
    hy = HybridExtractor(
        [HFEnergyExtractor(img_size=64, verbose=False), LPQExtractor(img_size=64, verbose=False)],
        names=["freq", "lpq"],
    )
    X = hy.transform(image_paths)
    names = hy.get_feature_names()
    assert names is not None
    assert len(names) == X.shape[1]
    assert names[0] == "freq__ring_00"
    assert sum(n.startswith("lpq__lpq_bin") for n in names) == 256


def test_fusion_pca_names(image_paths):
    fx = FeatureFusionExtractor(
        [HFEnergyExtractor(img_size=64, verbose=False)], pca_components=2, verbose=False
    )
    fx.fit(image_paths)
    X = fx.transform(image_paths)
    assert X.shape[1] == 2
    assert fx.get_feature_names() == ["pca_000", "pca_001"]


def test_fusion_vt_names(image_paths):
    fx = FeatureFusionExtractor(
        [HFEnergyExtractor(img_size=64, verbose=False)],
        names=["freq"],
        variance_threshold=0.0,
        verbose=False,
    )
    fx.fit(image_paths)
    X = fx.transform(image_paths)
    names = fx.get_feature_names()
    assert names is not None
    assert len(names) == X.shape[1]
    # имена после VT — подмножество имён hybrid (с префиксом части)
    hybrid_names = fx.hybrid_.get_feature_names()
    assert set(names) <= set(hybrid_names)
    assert all(n.startswith("freq__") for n in names)


def test_fusion_plain_names(image_paths):
    fx = FeatureFusionExtractor([HFEnergyExtractor(img_size=64, verbose=False)], verbose=False)
    fx.fit(image_paths)
    X = fx.transform(image_paths)
    names = fx.get_feature_names()
    assert names is not None
    assert len(names) == X.shape[1]
    assert names == fx.hybrid_.get_feature_names()


def test_mismatched_preset_names_are_reset(image_paths):
    """Несовпадение длины имён с размерностью — warning и сброс, а не падение."""
    ext = HFEnergyExtractor(img_size=64, verbose=False)
    ext.feature_names_ = ["only_one_wrong_name"]
    X = ext.transform(image_paths)
    assert X.shape[0] == 3
    assert ext.get_feature_names() is None
