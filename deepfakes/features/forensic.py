"""FOR-1: классическая форензика — Error Level Analysis (ELA).

Идея: пережимаем изображение в JPEG с известным качеством q и смотрим на
поканальную разницу |I - JPEG_q(I)|. У изображения, уже прошедшего JPEG-сжатие
(типичный «честный» снимок из интернета), повторное сжатие меняет пиксели
слабо и равномерно. Сгенерированные/отредактированные области (GAN/Diffusion,
вклейки) дают другой уровень и другую пространственную структуру ошибки —
это и ловят глобальные и патчевые статистики ELA-карты.
"""

from __future__ import annotations

import cv2
import numpy as np

from deepfakes.features.base import BaseFeatureExtractor
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


class ELAExtractor(BaseFeatureExtractor):
    """FOR-1: Error Level Analysis.

    Пайплайн: RGB-вход → JPEG-пережатие с качеством `quality` (cv2.imencode /
    imdecode) → diff = |img - recompressed| поканально (float).

    Признаки:
      * по каждому из 3 каналов глобально: mean, std, max, p95 (12 чисел);
      * по сетке patch_grid×patch_grid для каждого канала: mean, std
        (patch_grid² × 3 × 2 чисел).

    Параметры:
        img_size: размер ресайза входа.
        quality: JPEG-качество пережатия (default 90).
        patch_grid: сетка патчевых статистик (default 4).
    """

    def __init__(
        self,
        img_size: int = 256,
        quality: int = 90,
        patch_grid: int = 4,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=False, verbose=verbose)
        self.quality = int(quality)
        self.patch_grid = max(1, int(patch_grid))

    def _ela_map(self, img: np.ndarray) -> np.ndarray:
        """Карта ошибки пережатия |img - JPEG_q(img)|, float32, HxWx3 (RGB)."""
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            raise ValueError("cv2.imencode('.jpg') failed")
        rec_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        rec = cv2.cvtColor(rec_bgr, cv2.COLOR_BGR2RGB)
        return np.abs(img.astype(np.float32) - rec.astype(np.float32))

    def _extract(self, img: np.ndarray) -> np.ndarray:
        diff = self._ela_map(img)
        feats: list[float] = []
        # Глобальные статистики по каналам
        for c in range(diff.shape[2]):
            ch = diff[:, :, c]
            feats.extend(
                [
                    float(ch.mean()),
                    float(ch.std()),
                    float(ch.max()),
                    float(np.percentile(ch, 95)),
                ]
            )
        # Патчевые mean/std по сетке
        grid = self.patch_grid
        h, w = diff.shape[:2]
        bh, bw = h // grid, w // grid
        for gy in range(grid):
            for gx in range(grid):
                block = diff[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                for c in range(block.shape[2]):
                    ch = block[:, :, c]
                    feats.extend([float(ch.mean()), float(ch.std())])
        return np.asarray(feats, dtype=np.float32)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: глобальные по каналам R/G/B, затем патчи сетки."""
        chans = ("R", "G", "B")
        names = [f"global_{c}_{s}" for c in chans for s in ("mean", "std", "max", "p95")]
        for gy in range(self.patch_grid):
            for gx in range(self.patch_grid):
                for c in chans:
                    names.extend(f"g{gy}_{gx}_{c}_{s}" for s in ("mean", "std"))
        return names
