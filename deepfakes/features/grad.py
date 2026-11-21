"""Градиентные признаки: Sobel (GRAD-1), Scharr (GRAD-2), Canny (GRAD-3), LoG (GRAD-4)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from deepfakes.features.base import BaseFeatureExtractor


def _orientation_histogram(mag: np.ndarray, ang: np.ndarray, n_bins: int = 9) -> np.ndarray:
    """Взвешенная по магнитуде гистограмма ориентаций (0..π)."""
    ang = np.mod(ang, np.pi)
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    hist, _ = np.histogram(ang.ravel(), bins=bin_edges, weights=mag.ravel())
    s = hist.sum()
    return (hist / s).astype(np.float32) if s > 0 else hist.astype(np.float32)


def _patch_stats(arr: np.ndarray) -> list[float]:
    v = arr.reshape(-1).astype(np.float64)
    if v.std() < 1e-8:
        return [float(v.mean()), 0.0, 0.0, 0.0]
    return [
        float(v.mean()),
        float(v.std()),
        float(kurtosis(v)),
        float(skew(v)),
    ]


#: имена статистик _patch_stats (порядок как в возвращаемом списке)
_PATCH_STAT_NAMES = ("mean", "std", "kurt", "skew")


def _gradient_patch_feature_names(patch_grid: int, n_orient_bins: int) -> list[str]:
    """Имена признаков _gradient_patch_features (общая схема GRAD-1/GRAD-2)."""
    names: list[str] = []
    for gy in range(patch_grid):
        for gx in range(patch_grid):
            p = f"g{gy}_{gx}"
            names.extend(f"{p}_mag_{s}" for s in _PATCH_STAT_NAMES)
            names.extend(f"{p}_orient_bin{i}" for i in range(n_orient_bins))
    names += ["global_mag_mean", "global_mag_std", "global_mag_frac_above_mean"]
    return names


def _gradient_patch_features(
    mag: np.ndarray,
    ang: np.ndarray,
    patch_grid: int,
    n_orient_bins: int,
    threshold: float | None,
) -> np.ndarray:
    """Общая схема признаков GRAD-1/GRAD-2: патчевые статистики магнитуды +
    взвешенные магнитудой гистограммы ориентаций + глобальные статистики."""
    if threshold is not None:
        mag = np.where(mag >= threshold, mag, 0.0)
    h, w = mag.shape
    bh, bw = h // patch_grid, w // patch_grid
    feats: list[float] = []
    for gy_idx in range(patch_grid):
        for gx_idx in range(patch_grid):
            mblock = mag[gy_idx * bh : (gy_idx + 1) * bh, gx_idx * bw : (gx_idx + 1) * bw]
            ablock = ang[gy_idx * bh : (gy_idx + 1) * bh, gx_idx * bw : (gx_idx + 1) * bw]
            feats.extend(_patch_stats(mblock))
            feats.extend(_orientation_histogram(mblock, ablock, n_orient_bins).tolist())
    # глобальные
    feats.extend(
        [
            float(mag.mean()),
            float(mag.std()),
            float((mag > mag.mean()).mean()),
        ]
    )
    return np.asarray(feats, dtype=np.float32)


class SobelExtractor(BaseFeatureExtractor):
    """GRAD-1: Sobel magnitude + orientation histograms per patch.

    Из таблицы: «Sobel, Spatial aug, noise aug, SVM (RBF)». Это компактный
    counterpart GRAD-4 (LoG). Возвращает (n_patches × (4 + n_orient_bins)) признаков.
    """

    def __init__(
        self,
        img_size: int = 128,
        ksize: int = 3,
        patch_grid: int = 4,
        n_orient_bins: int = 9,
        threshold: float | None = None,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.ksize = ksize
        self.patch_grid = max(1, int(patch_grid))
        self.n_orient_bins = n_orient_bins
        self.threshold = threshold

    def _extract(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=self.ksize)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=self.ksize)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.arctan2(gy, gx)
        return _gradient_patch_features(
            mag, ang, self.patch_grid, self.n_orient_bins, self.threshold
        )

    def _build_feature_names(self) -> list[str]:
        return _gradient_patch_feature_names(self.patch_grid, self.n_orient_bins)


class ScharrExtractor(BaseFeatureExtractor):
    """GRAD-2: Scharr magnitude + orientation histograms per patch.

    Зеркальный аналог GRAD-1 (Sobel): ядро Scharr 3×3 точнее аппроксимирует
    производную и ротационно устойчивее, чем Sobel 3×3 (размер ядра у Scharr
    фиксирован, поэтому параметра ksize нет). Набор признаков идентичен GRAD-1:
    n_patches × (4 + n_orient_bins) патчевых + 3 глобальных.
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_grid: int = 4,
        n_orient_bins: int = 9,
        threshold: float | None = None,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.patch_grid = max(1, int(patch_grid))
        self.n_orient_bins = n_orient_bins
        self.threshold = threshold

    def _extract(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.arctan2(gy, gx)
        return _gradient_patch_features(
            mag, ang, self.patch_grid, self.n_orient_bins, self.threshold
        )

    def _build_feature_names(self) -> list[str]:
        return _gradient_patch_feature_names(self.patch_grid, self.n_orient_bins)


class CannyEdgeExtractor(BaseFeatureExtractor):
    """GRAD-3: Canny edges → edge density per patch + orientation histogram + global stats.

    Из таблицы: «Canny + этап построения гистограммы от контуров Canny, RF, TTA, RF importance,
    GAN + JPEG, артефакты компрессии».
    """

    def __init__(
        self,
        img_size: int = 256,
        low_threshold: int = 50,
        high_threshold: int = 150,
        aperture_size: int = 3,
        patch_grid: int = 4,
        n_orient_bins: int = 9,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)
        self.aperture_size = int(aperture_size)
        self.patch_grid = max(1, int(patch_grid))
        self.n_orient_bins = n_orient_bins

    def _extract(self, img: np.ndarray) -> np.ndarray:
        # L2gradient=True — каноническая L2-магнитуда градиента (sqrt(gx^2 + gy^2))
        # вместо дефолтной L1-аппроксимации |gx| + |gy|.
        edges = cv2.Canny(
            img,
            self.low_threshold,
            self.high_threshold,
            apertureSize=self.aperture_size,
            L2gradient=True,
        )
        # orientations via Sobel of the original image (на самих edges градиенты не информативны)
        gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        ang = np.arctan2(gy, gx)
        mag = np.sqrt(gx * gx + gy * gy)
        mag_at_edges = np.where(edges > 0, mag, 0.0)

        h, w = edges.shape
        bh, bw = h // self.patch_grid, w // self.patch_grid
        feats: list[float] = []
        for gy_idx in range(self.patch_grid):
            for gx_idx in range(self.patch_grid):
                e_block = edges[gy_idx * bh : (gy_idx + 1) * bh, gx_idx * bw : (gx_idx + 1) * bw]
                a_block = ang[gy_idx * bh : (gy_idx + 1) * bh, gx_idx * bw : (gx_idx + 1) * bw]
                m_block = mag_at_edges[
                    gy_idx * bh : (gy_idx + 1) * bh, gx_idx * bw : (gx_idx + 1) * bw
                ]
                feats.append(float((e_block > 0).mean()))  # edge density
                feats.append(float(m_block.mean()))  # mean magnitude on edges
                feats.extend(_orientation_histogram(m_block, a_block, self.n_orient_bins).tolist())
        feats.extend(
            [
                float((edges > 0).mean()),  # global edge density
                float(np.count_nonzero(edges)),
            ]
        )
        return np.asarray(feats, dtype=np.float32)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: патч → density, mag_mean, orient-бины; затем глобальные."""
        names: list[str] = []
        for gy in range(self.patch_grid):
            for gx in range(self.patch_grid):
                p = f"g{gy}_{gx}"
                names.append(f"{p}_edge_density")
                names.append(f"{p}_mag_mean")
                names.extend(f"{p}_orient_bin{i}" for i in range(self.n_orient_bins))
        names += ["global_edge_density", "global_edge_count"]
        return names


def _log_zero_crossings(
    img: np.ndarray, sigma: float = 0.7, zc_threshold: float = 0.5
) -> np.ndarray:
    """LoG (Laplacian of Gaussian) карта zero-crossings, нормализованная.

    Marr–Hildreth требует порога на силу перепада: без него смена знака LoG
    возникает почти в каждом пикселе из-за шума. Переход соседних значений a, b
    засчитывается как zero-crossing, только если знаки различаются И локальный
    перепад |a − b| превышает zc_threshold * std(LoG-карты). Нули LoG считаются
    положительными (иначе np.sign(0) == 0 давал бы ложные «смены знака» вокруг
    плоских областей).
    """
    img_f = img.astype(np.float32)
    if img_f.max() > 1.5:
        img_f = img_f / 255.0
    blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
    log_ = cv2.Laplacian(blur, cv2.CV_32F)
    # нули считаем положительными: sign(0)=0 ломал бы сравнение знаков
    s = np.where(log_ >= 0, 1.0, -1.0).astype(np.float32)
    thr = float(zc_threshold) * float(log_.std())
    diff_h = (s[:, 1:] != s[:, :-1]) & (np.abs(log_[:, 1:] - log_[:, :-1]) > thr)
    diff_v = (s[1:, :] != s[:-1, :]) & (np.abs(log_[1:, :] - log_[:-1, :]) > thr)
    zc = np.zeros_like(log_)
    zc[:, 1:] += diff_h
    zc[1:, :] += diff_v
    return (zc > 0).astype(np.float32)


class LoGExtractor(BaseFeatureExtractor):
    """GRAD-4 переписанный: вместо flatten + XGBoost — patch-aggregation.

    Разбить LoG-карту на патчи, по каждому
    считать статистики (edge_density, mean, std, kurtosis, skew). Размерность падает
    с 16k+ до ~ (img_size/patch)^2 * 5.

    Default sigma поднят 0.7 → 1.2: при 0.7 сглаживание почти не подавляло
    пиксельный шум, и карта zero-crossings определялась в основном шумом,
    а не структурой изображения.
    """

    def __init__(
        self,
        img_size: int = 128,
        sigma: float = 1.2,
        zc_threshold: float = 0.5,
        patch_size: int = 16,
        full_stats: bool = True,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.sigma = sigma
        self.zc_threshold = zc_threshold
        self.patch_size = patch_size
        self.full_stats = full_stats

    def _extract(self, img: np.ndarray) -> np.ndarray:
        zc = _log_zero_crossings(img, sigma=self.sigma, zc_threshold=self.zc_threshold)
        h, w = zc.shape
        ps = self.patch_size
        feats: list[float] = []
        for y in range(0, h, ps):
            for x in range(0, w, ps):
                if y + ps > h or x + ps > w:
                    continue
                patch = zc[y : y + ps, x : x + ps].reshape(-1)
                edge_density = float(patch.mean())
                feats.append(edge_density)
                if self.full_stats:
                    feats.append(float(patch.std()))
                    feats.append(float(kurtosis(patch)) if patch.std() > 1e-8 else 0.0)
                    feats.append(float(skew(patch)) if patch.std() > 1e-8 else 0.0)
        # Глобальные статистики
        feats.extend(
            [
                float(zc.mean()),
                float(zc.std()),
                float((zc > 0).sum() / zc.size),
            ]
        )
        return np.asarray(feats, dtype=np.float32)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: патчи LoG-карты (строго те же циклы) + глобальные."""
        names: list[str] = []
        size, ps = self.img_size, self.patch_size
        for y in range(0, size, ps):
            for x in range(0, size, ps):
                if y + ps > size or x + ps > size:
                    continue
                p = f"p{y // ps}_{x // ps}"
                names.append(f"{p}_edge_density")
                if self.full_stats:
                    names.extend([f"{p}_std", f"{p}_kurt", f"{p}_skew"])
        names += ["global_mean", "global_std", "global_density"]
        return names


class LoGFlattenExtractor(BaseFeatureExtractor):
    """Оригинальный GRAD-4: flatten LoG-карты. Оставлен как honest baseline."""

    def __init__(
        self,
        img_size: int = 128,
        sigma: float = 0.7,
        zc_threshold: float = 0.5,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.sigma = sigma
        self.zc_threshold = zc_threshold

    def _extract(self, img: np.ndarray) -> np.ndarray:
        zc = _log_zero_crossings(img, sigma=self.sigma, zc_threshold=self.zc_threshold)
        return zc.ravel().astype(np.float32)

    def _build_feature_names(self) -> list[str]:
        """Flatten-режим: осмысленных имён нет — плейсхолдеры f000..."""
        n = self.n_features_out_ if self.n_features_out_ is not None else self.img_size**2
        return self._generic_names(n)
