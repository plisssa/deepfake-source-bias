"""Контроль «только предобработка»: симуляция DFF без генерации."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from scripts.processing_control import VARIANTS, original_stem, simulate


def _raw(w=440, h=660):
    arr = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="JPEG", quality=95)
    return b.getvalue()


def test_resize_variants_give_dff_geometry_and_quality():
    for name, rs in VARIANTS.items():
        im = Image.open(io.BytesIO(simulate(_raw(), rs)))
        expected = (440, 660) if rs is None else (512, 512)
        assert im.size == expected, name
        assert im.format == "JPEG"


def test_original_stem_strips_normalizer_index():
    assert original_stem("016477_1465655_1984-09-09_2014.jpg") == "1465655_1984-09-09_2014"
