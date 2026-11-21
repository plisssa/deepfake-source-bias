"""Пространственные аугментации: горизонтальный flip и random crop + resize.

Принимают и возвращают uint8; детерминизм — через ``np.random.Generator``.
"""

from __future__ import annotations

import cv2
import numpy as np


def hflip(img: np.ndarray) -> np.ndarray:
    """Горизонтальное отражение (детерминированное, rng не нужен)."""
    return np.ascontiguousarray(img[:, ::-1])


def random_crop_resize(
    img: np.ndarray, scale: float = 0.8, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Случайный кроп доли scale по каждой стороне + resize к исходному размеру."""
    rng = rng if rng is not None else np.random.default_rng()
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    h, w = img.shape[:2]
    ch = max(1, round(h * float(scale)))
    cw = max(1, round(w * float(scale)))
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    crop = img[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
