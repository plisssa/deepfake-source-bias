"""Post-processing perturbations for robustness evaluation: JPEG, resize, blur and noise."""

from __future__ import annotations

import cv2
import numpy as np


def jpeg_recompress(img: np.ndarray, rng: np.random.Generator | None = None, quality: int = 75):
    """Пережать JPEG с заданным качеством (самая частая деградация в вебе)."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    return out if out is not None else img


def downscale_upscale(img: np.ndarray, rng: np.random.Generator | None = None, scale: float = 0.5):
    """Уменьшить и вернуть к исходному размеру — убивает высокие частоты.

    Именно этот канал (апскейл-асимметрия) испортил прогон 30.07.2026, поэтому
    здесь он используется осознанно: как контролируемая деградация ОБОИХ классов."""
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def gaussian_blur(img: np.ndarray, rng: np.random.Generator | None = None, sigma: float = 1.0):
    """Гауссово размытие (мягкая потеря деталей)."""
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma))


def additive_noise(img: np.ndarray, rng: np.random.Generator | None = None, sigma: float = 5.0):
    """Аддитивный гауссов шум (имитация слабого сенсорного/канального шума)."""
    r = rng if rng is not None else np.random.default_rng(0)
    noisy = img.astype(np.float32) + r.normal(0.0, float(sigma), img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


#: сценарии стенда: имя → (функция, параметры). Имя попадает в метрику MLflow.
DEGRADATIONS: dict[str, tuple] = {
    "clean": (lambda img, rng=None: img, {}),
    "jpeg90": (jpeg_recompress, {"quality": 90}),
    "jpeg75": (jpeg_recompress, {"quality": 75}),
    "jpeg50": (jpeg_recompress, {"quality": 50}),
    "scale75": (downscale_upscale, {"scale": 0.75}),
    "scale50": (downscale_upscale, {"scale": 0.5}),
    "blur1": (gaussian_blur, {"sigma": 1.0}),
    "noise5": (additive_noise, {"sigma": 5.0}),
}
