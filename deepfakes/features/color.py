"""Цветовые признаки: multi-scale color moments + расширенный набор статистик."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import entropy, kurtosis, skew

from deepfakes.features.base import BaseFeatureExtractor


def _patch_stats(
    patch: np.ndarray, eps: float = 1e-8, value_range: tuple[int, int] = (0, 256)
) -> list[float]:
    """Расширенный набор статистик по одному каналу патча.

    В число признаков входят kurtosis и entropy.
    value_range — фактический диапазон значений канала (для гистограммы энтропии).
    """
    p = patch.reshape(-1).astype(np.float64)
    if p.size == 0:
        return [0.0] * 9
    mean = float(np.mean(p))
    std = float(np.std(p))
    sk = float(skew(p)) if p.std() > eps else 0.0
    kt = float(kurtosis(p)) if p.std() > eps else 0.0
    # Энтропия по 32-бин гистограмме, нормировка явная — на сумму бинов
    # (density=True нормировала бы на ширину бина, а не в распределение)
    hist, _ = np.histogram(p, bins=32, range=value_range)
    hist = hist.astype(np.float64)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    hist = hist + eps
    ent = float(entropy(hist))
    p10, p50, p90 = np.percentile(p, [10, 50, 90])
    iqr = float(np.percentile(p, 75) - np.percentile(p, 25))
    return [mean, std, sk, kt, ent, float(p10), float(p50), float(p90), iqr]


def _circular_mean_std_hue(p: np.ndarray) -> tuple[float, float]:
    """Циркулярные mean/std для канала H (OpenCV uint8: H ∈ 0..180, шаг 2°).

    Угол = H * 2 градусов → радианы; circular mean — через atan2 средних
    sin/cos, результат переводится обратно в единицы H (0..180);
    circular std = sqrt(−2 ln R) (в радианах), R — длина среднего вектора.
    """
    ang = np.deg2rad(p.reshape(-1).astype(np.float64) * 2.0)
    sin_m = float(np.sin(ang).mean())
    cos_m = float(np.cos(ang).mean())
    mean_h = float(np.rad2deg(np.arctan2(sin_m, cos_m)) / 2.0 % 180.0)
    r = min(1.0, float(np.hypot(sin_m, cos_m)))
    # R → 0 (равномерное распределение) даёт std → inf; ограничиваем снизу
    circ_std = float(np.sqrt(-2.0 * np.log(max(r, 1e-12))))
    return mean_h, circ_std


def _batch_patch_stats(
    p: np.ndarray,
    value_range: tuple[int, int],
    full_stats: bool = True,
    circular: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """Статистики _patch_stats для ВСЕХ патчей канала разом: p (n_patches, npix).

    Возвращает (n_patches, 9) при full_stats (или (n, 3) без) в том же порядке,
    что и поэлементный _patch_stats: mean, std, skew, kurt, entropy, p10, p50,
    p90, iqr. Формулы идентичны scipy (bias=True), включая обнуление
    skew/kurt при std<=eps."""
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import skew as _skew

    n, _npix = p.shape
    mean = p.mean(axis=1)
    std = p.std(axis=1)
    ok = std > eps
    sk = np.where(ok, np.nan_to_num(_skew(p, axis=1)), 0.0)
    if not full_stats:
        stats = np.stack([mean, std, sk], axis=1)
    else:
        kt = np.where(ok, np.nan_to_num(_kurt(p, axis=1)), 0.0)
        # энтропия: биннинг вручную (эквивалент np.histogram на равномерных
        # бинах в пределах value_range; значения uint8 всегда внутри диапазона)
        lo, hi = value_range
        idx = np.floor((p - lo) * (32.0 / (hi - lo))).astype(np.int64)
        np.clip(idx, 0, 31, out=idx)
        flat = (np.arange(n)[:, None] * 32 + idx).reshape(-1)
        hist = np.bincount(flat, minlength=n * 32).reshape(n, 32).astype(np.float64)
        sums = hist.sum(axis=1, keepdims=True)
        hist = np.divide(hist, sums, out=np.zeros_like(hist), where=sums > 0)
        hist = hist + eps
        pk = hist / hist.sum(axis=1, keepdims=True)
        ent = -(pk * np.log(pk)).sum(axis=1)
        p10, p25, p50, p75, p90 = np.percentile(p, [10, 25, 50, 75, 90], axis=1)
        iqr = p75 - p25
        stats = np.stack([mean, std, sk, kt, ent, p10, p50, p90, iqr], axis=1)
    if circular and p.shape[1] > 0:
        ang = np.deg2rad(p * 2.0)
        sin_m = np.sin(ang).mean(axis=1)
        cos_m = np.cos(ang).mean(axis=1)
        stats[:, 0] = np.rad2deg(np.arctan2(sin_m, cos_m)) / 2.0 % 180.0
        r = np.minimum(1.0, np.hypot(sin_m, cos_m))
        stats[:, 1] = np.sqrt(-2.0 * np.log(np.maximum(r, 1e-12)))
    return stats


_COLORSPACE_CVT = {
    "RGB": None,
    "HSV": cv2.COLOR_RGB2HSV,
    "YCbCr": cv2.COLOR_RGB2YCrCb,
    "Lab": cv2.COLOR_RGB2LAB,
}

#: имена каналов по colorspace — для имён признаков
_CHANNEL_NAMES = {
    "RGB": ("R", "G", "B"),
    "HSV": ("H", "S", "V"),
    "YCbCr": ("Y", "Cb", "Cr"),
    "Lab": ("L", "a", "b"),
}

#: имена статистик _patch_stats (порядок как в возвращаемом списке)
_PATCH_STAT_NAMES = ("mean", "std", "skew", "kurt", "entropy", "p10", "p50", "p90", "iqr")


def _convert_colorspace(img: np.ndarray, colorspace: str) -> np.ndarray:
    """RGB → указанное пространство; для 'YCbCr' каналы приводятся к (Y, Cb, Cr).

    В OpenCV есть только COLOR_RGB2YCrCb (порядок Y, Cr, Cb) — переставляем
    каналы, чтобы семантика соответствовала названию ключа 'YCbCr'."""
    cvt = _COLORSPACE_CVT.get(colorspace)
    out = img if cvt is None else cv2.cvtColor(img, cvt)
    if colorspace == "YCbCr":
        out = out[..., [0, 2, 1]]
    return out


def _channel_ranges(colorspace: str) -> list[tuple[int, int]]:
    """Фактические диапазоны значений каналов для гистограмм.

    В OpenCV (uint8) канал H пространства HSV лежит в 0..180,
    все остальные каналы всех пространств — в 0..256."""
    if colorspace == "HSV":
        return [(0, 180), (0, 256), (0, 256)]
    return [(0, 256)] * 3


class ColorHistogramExtractor(BaseFeatureExtractor):
    """COL-1: канальные гистограммы (RGB / HSV / YCbCr / Lab), нормированные на сумму.

    Очень простой baseline. Возвращает (n_channels × n_bins) признаков.
    Гистограммы считаются в фактическом диапазоне канала (для H в HSV — 0..180)
    и нормируются явно на сумму бинов (sum = 1), а не через density=True
    (та делит на ширину бина). Дополнительно умеет резать сетку
    patch_grid×patch_grid и считать гистограммы локально — это превращает COL-1
    в spatial-aware вариант.
    """

    def __init__(
        self,
        img_size: int = 256,
        n_bins: int = 32,
        colorspace: str = "RGB",
        patch_grid: int = 1,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=False, verbose=verbose)
        self.n_bins = int(n_bins)
        self.colorspace = colorspace
        self.patch_grid = max(1, int(patch_grid))

    def _convert(self, img: np.ndarray) -> np.ndarray:
        return _convert_colorspace(img, self.colorspace)

    def _hist(self, arr: np.ndarray, value_range: tuple[int, int]) -> np.ndarray:
        h, _ = np.histogram(arr.ravel(), bins=self.n_bins, range=value_range)
        h = h.astype(np.float64)
        s = h.sum()
        return (h / s if s > 0 else h).astype(np.float32)

    def _extract(self, img: np.ndarray) -> np.ndarray:
        img = self._convert(img)
        h, w = img.shape[:2]
        channels = cv2.split(img)
        ranges = _channel_ranges(self.colorspace)
        feats: list[float] = []
        if self.patch_grid == 1:
            for c, rng in zip(channels, ranges):
                feats.extend(self._hist(c, rng).tolist())
        else:
            grid = self.patch_grid
            bh, bw = h // grid, w // grid
            for gy in range(grid):
                for gx in range(grid):
                    for c, rng in zip(channels, ranges):
                        block = c[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                        feats.extend(self._hist(block, rng).tolist())
        return np.asarray(feats, dtype=np.float32)

    def _build_feature_names(self) -> list[str] | None:
        """Имена в порядке _extract: (патч сетки) → канал → бины гистограммы."""
        chans = _CHANNEL_NAMES.get(self.colorspace)
        if chans is None:
            return None
        bins = [f"bin{i:02d}" for i in range(self.n_bins)]
        names: list[str] = []
        if self.patch_grid == 1:
            for c in chans:
                names.extend(f"{c}_{b}" for b in bins)
            return names
        for gy in range(self.patch_grid):
            for gx in range(self.patch_grid):
                for c in chans:
                    names.extend(f"g{gy}_{gx}_{c}_{b}" for b in bins)
        return names


def _chi_square(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Chi-square distance между двумя нормированными гистограммами."""
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + eps)))


class YCbCrChiSquareExtractor(BaseFeatureExtractor):
    """COL-3: YCbCr (Cb/Cr) — color moments + chi-square distance local vs global.

    Дополняет COL-2 фокусом на хроматических каналах: моменты по Cb и Cr,
    плюс chi-square расстояния между локальной гистограммой каждого патча
    и глобальной гистограммой канала по всему изображению.
    """

    def __init__(
        self,
        img_size: int = 256,
        n_bins: int = 32,
        patch_grid: int = 4,
        full_stats: bool = True,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=False, verbose=verbose)
        self.n_bins = int(n_bins)
        self.patch_grid = max(2, int(patch_grid))
        self.full_stats = full_stats

    def _hist(self, arr: np.ndarray) -> np.ndarray:
        # нормировка явная — на сумму бинов (не density=True: та делит на ширину бина)
        h, _ = np.histogram(arr.ravel(), bins=self.n_bins, range=(0, 256))
        h = h.astype(np.float64)
        s = h.sum()
        return h / s if s > 0 else h

    def _moments(self, arr: np.ndarray) -> list[float]:
        return (
            _patch_stats(arr)
            if self.full_stats
            else [
                float(np.mean(arr)),
                float(np.std(arr)),
                float(skew(arr.reshape(-1))) if arr.std() > 1e-8 else 0.0,
            ]
        )

    def _extract(self, img: np.ndarray) -> np.ndarray:
        # единый порядок каналов (Y, Cb, Cr) через общий конвертер
        ycbcr = _convert_colorspace(img, "YCbCr")
        _, cb, cr = cv2.split(ycbcr)
        feats: list[float] = []
        h, w = cr.shape
        grid = self.patch_grid
        bh, bw = h // grid, w // grid
        for chan in (cb, cr):
            # глобальные моменты + глобальная гистограмма
            feats.extend(self._moments(chan))
            global_hist = self._hist(chan)
            # локальные моменты + chi-square distance к global
            for gy in range(grid):
                for gx in range(grid):
                    block = chan[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                    feats.extend(self._moments(block))
                    feats.append(_chi_square(self._hist(block), global_hist))
        return np.asarray(feats, dtype=np.float32)


class ColorMomentsExtractor(BaseFeatureExtractor):
    """COL-2 переписанный: multi-scale + расширенные статистики + выбор colorspace.

    Параметры:
        patch_sizes: список размеров патчей, например (8, 16, 32). Для каждого размера
            строится сетка патчей, по каждому каналу считается полный набор статистик,
            и всё конкатенируется.
        colorspace: 'RGB' | 'HSV' | 'YCbCr' | 'Lab'. Для 'HSV' по каналу H
            mean/std заменяются на циркулярные (H — угловая величина: значения
            179 и 1 соседние, обычные mean/std на границе дают ~90 и ложный
            разброс); остальные статистики и остальные каналы без изменений.
        full_stats: если False — только (mean, std, skew) как в оригинальном COL-2.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_sizes: tuple[int, ...] = (32,),
        colorspace: str = "RGB",
        full_stats: bool = True,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=False, verbose=verbose)
        self.patch_sizes = tuple(patch_sizes)
        self.colorspace = colorspace
        self.full_stats = full_stats

    def _convert(self, img: np.ndarray) -> np.ndarray:
        return _convert_colorspace(img, self.colorspace)

    def _stats(
        self,
        patch: np.ndarray,
        value_range: tuple[int, int] = (0, 256),
        circular: bool = False,
    ) -> list[float]:
        p = patch.reshape(-1).astype(np.float64)
        if self.full_stats:
            stats = _patch_stats(patch, value_range=value_range)
        else:
            stats = [
                float(np.mean(p)),
                float(np.std(p)),
                float(skew(p)) if p.std() > 1e-8 else 0.0,
            ]
        if circular and p.size > 0:
            stats[0], stats[1] = _circular_mean_std_hue(p)
        return stats

    def _extract(self, img: np.ndarray) -> np.ndarray:
        """Векторизованная версия: все патчи одного масштаба считаются разом.

        Старая реализация делала Python-вызов статистик на КАЖДЫЙ (патч, канал)
        — для 256px и патчей (8,16,32) это 4032 вызова со scipy-накладными;
        на кластерных Xeon это давало ~0.6 с/изображение. Здесь патчи
        собираются reshape-ом в (n_patches, ps*ps), и каждая статистика —
        одна numpy-операция по оси. Порядок признаков идентичен старому:
        ps → y → x → канал → статистики (проверяется тестом эквивалентности).
        """
        img = self._convert(img)
        h, w = img.shape[:2]
        ranges = _channel_ranges(self.colorspace)
        n_stats = 9 if self.full_stats else 3
        out: list[np.ndarray] = []
        for ps in self.patch_sizes:
            gy, gx = h // ps, w // ps
            if gy == 0 or gx == 0:
                continue
            crop = img[: gy * ps, : gx * ps].astype(np.float64)
            # (gy, ps, gx, ps, 3) → (gy*gx, ps*ps, 3): патчи в порядке y-major
            patches = (
                crop.reshape(gy, ps, gx, ps, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(gy * gx, ps * ps, 3)
            )
            # (n, 3, n_stats) — статистики всех патчей всех каналов
            block = np.empty((gy * gx, 3, n_stats), dtype=np.float64)
            for ci in range(3):
                p = patches[:, :, ci]  # (n_patches, ps*ps)
                circular = self.colorspace == "HSV" and ci == 0
                block[:, ci, :] = _batch_patch_stats(
                    p,
                    value_range=ranges[ci],
                    full_stats=self.full_stats,
                    circular=circular,
                )
            out.append(block.reshape(-1))
        return np.concatenate(out).astype(np.float32)

    def _build_feature_names(self) -> list[str] | None:
        """Имена в порядке _extract: patch_size → (y, x) патча → канал → статистики."""
        chans = _CHANNEL_NAMES.get(self.colorspace)
        if chans is None:
            return None
        stats = _PATCH_STAT_NAMES if self.full_stats else _PATCH_STAT_NAMES[:3]
        size = self.img_size
        names: list[str] = []
        for ps in self.patch_sizes:
            for y in range(0, size, ps):
                for x in range(0, size, ps):
                    if y + ps > size or x + ps > size:
                        continue
                    for c in chans:
                        names.extend(f"patch{ps}_y{y}_x{x}_ch{c}_{st}" for st in stats)
        return names
