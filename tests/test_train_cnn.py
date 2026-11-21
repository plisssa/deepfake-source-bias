"""CNN-бейзлайн: вход без интерполяции и аугментация Wang et al. (без torch)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.train_cnn import blur_jpeg, crop, variant_name


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (256, 256, 3), dtype=np.uint8)


def test_eval_crop_is_center_and_pixel_exact():
    img = _img()
    np.testing.assert_array_equal(crop(img, 224, train=False, rng=None), img[16:240, 16:240])


def test_train_crop_stays_inside_and_varies():
    img = _img()
    rng = np.random.default_rng(1)
    outs = [crop(img, 224, train=True, rng=rng) for _ in range(20)]
    assert all(o.shape == (224, 224, 3) for o in outs)
    assert len({o.tobytes() for o in outs}) > 1, "случайный кроп должен меняться"


def test_crop_rejects_too_small():
    with pytest.raises(ValueError):
        crop(np.zeros((200, 200, 3), np.uint8), 224, train=False, rng=None)


def test_blur_jpeg_shape_dtype_and_determinism():
    img = _img(2)
    a = blur_jpeg(img, np.random.default_rng(5), p=1.0)
    b = blur_jpeg(img, np.random.default_rng(5), p=1.0)
    assert a.shape == img.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, img)


def test_blur_jpeg_p0_is_identity():
    img = _img(3)
    np.testing.assert_array_equal(blur_jpeg(img, np.random.default_rng(0), p=0.0), img)


def test_variant_names_are_distinct():
    assert variant_name("resnet50.tv_in1k", "none") == "resnet50_ft"
    assert variant_name("resnet50.tv_in1k", "blur_jpeg") == "resnet50_ft_blur_jpeg"
