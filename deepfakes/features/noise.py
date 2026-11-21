"""NOISE-1: noise residual + SRM features.

Идея: реальные камеры оставляют сенсорный шум с характерной статистикой;
GAN/Diffusion-генераторы оставляют другой «шумовой отпечаток». Поэтому
после денойзинга residual (img - denoised) у real и fake различается.

Реализованы два независимых источника признаков:
  1. Noise residual через денойзер (NLM из OpenCV по умолчанию; опционально BM3D
     через `bm3d` если установлен).
  2. SRM (Spatial Rich Model) — фиксированные high-pass ядра из стеганоанализа
     (Fridrich & Kodovsky, 2012). Применяем 6 ядер, по каждой residual-карте
     считаем расширенный набор статистик.

Закрывает пункт NOISE-1 из таблицы пайплайнов.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from deepfakes.features.base import BaseFeatureExtractor
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


# ---------- SRM high-pass kernels ----------


def _srm_kernels() -> list[np.ndarray]:
    """6 high-pass ядер в духе SRM (Fridrich & Kodovsky, 2012).

    Фактический состав (см. код ниже):
      1. KV 5×5 (нормировка /12);
      2-3. два сглаженных second-order ядра 3×3 (вертикальное и горизонтальное,
           нормировка /4);
      4-5. два линейных first-order ядра (разности соседей по столбцу и по строке);
      6. одно линейное second-order ядро [1, -2, 1] (по строке).
    """
    k = []
    # KV (5x5)
    k.append(
        np.array(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=np.float32,
        )
        / 12.0
    )
    # Smoothed second-order 3x3 (vertical)
    k.append(np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]], dtype=np.float32) / 4.0)
    # Smoothed second-order 3x3 (horizontal)
    k.append(np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]], dtype=np.float32) / 4.0)
    # 1st-order linear (col)
    k.append(np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32))
    # 1st-order linear (row)
    k.append(np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32))
    # 2nd-order linear (col)
    k.append(np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32))
    return k


#: имена статистик _residual_stats (порядок как в возвращаемом списке)
_RESIDUAL_STAT_NAMES = ("mean", "std", "skew", "kurt", "median", "p10_90", "abs_mean")


def _residual_stats(arr: np.ndarray, eps: float = 1e-8) -> list[float]:
    """Расширенный набор статистик по residual-карте."""
    v = arr.reshape(-1).astype(np.float64)
    if v.std() < eps:
        return [float(v.mean()), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(v.mean()),
        float(v.std()),
        float(skew(v)),
        float(kurtosis(v)),
        float(np.median(v)),
        float(np.percentile(v, 90) - np.percentile(v, 10)),
        float(np.mean(np.abs(v))),  # MAD-like
    ]


DenoiseMethod = Literal["nlmeans", "median", "bm3d"]
ResizeMode = Literal["center_crop", "resize"]


def _center_crop_or_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Центральный кроп size×size БЕЗ интерполяции; если меньше — паддинг BORDER_REFLECT.

    Зачем: cv2.resize с интерполяцией усредняет соседние пиксели и тем самым
    разрушает пиксельно-выровненные низкоуровневые артефакты — 8×8 сетку JPEG
    (границы блоков «размазываются» и перестают попадать на кратные 8 позиции)
    и высокочастотный сенсорный шум (PRNU), который живёт на масштабе 1 пикселя.
    Кроп сохраняет исходные значения пикселей и их решётку как есть.
    """
    h, w = img.shape[:2]
    if h < size or w < size:
        pad_v = max(0, size - h)
        pad_h = max(0, size - w)
        top, bottom = pad_v // 2, pad_v - pad_v // 2
        left, right = pad_h // 2, pad_h - pad_h // 2
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size]


class _CenterCropReadMixin:
    """Чтение изображения с resize_mode='center_crop'|'resize' (grayscale-экстракторы).

    При 'center_crop' изображение не ресайзится: берётся центральное окно
    img_size×img_size (с рефлект-паддингом при нехватке) — см. _center_crop_or_pad.
    При 'resize' — поведение базового класса (cv2.resize, INTER_AREA).
    """

    def _read_image(self, path: str) -> np.ndarray:
        if self.resize_mode == "resize":
            return super()._read_image(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Can't read image: {path}")
        return _center_crop_or_pad(img, self.img_size)

    def _prepare_raw(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.resize_mode == "resize":
            return super()._prepare_raw(img_bgr)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return _center_crop_or_pad(img, self.img_size)


def _denoise(
    img_gray: np.ndarray, method: DenoiseMethod = "nlmeans", strength: float = 10.0
) -> np.ndarray:
    """Снять шум одним из доступных способов."""
    if method == "nlmeans":
        return cv2.fastNlMeansDenoising(
            img_gray, None, h=strength, templateWindowSize=7, searchWindowSize=21
        )
    if method == "median":
        ksize = max(3, int(strength) | 1)
        return cv2.medianBlur(img_gray, ksize)
    if method == "bm3d":
        try:
            import bm3d
        except ImportError:
            log.warning("bm3d not installed, falling back to NLM")
            return cv2.fastNlMeansDenoising(img_gray, None, h=strength)
        return bm3d.bm3d(img_gray.astype(np.float32) / 255.0, sigma_psd=strength / 255.0) * 255.0
    raise ValueError(f"Unknown denoise method: {method}")


class JPEGBlockinessExtractor(_CenterCropReadMixin, BaseFeatureExtractor):
    """NOISE-2: мера блочности JPEG (8×8 сетка).

    Идея: пиксели на границах 8×8 блоков скачут сильнее, чем внутри. Считаем
    среднюю горизонтальную/вертикальную разность на границах vs внутри блока
    и их относительную разницу (упрощённая относительная блочность, НЕ
    каноническая метрика Wang–Bovik). GAN/Diffusion-генерации часто оставляют
    свою сетку или, наоборот, чрезмерно сглаживают границы — то и другое
    отделимо от честного JPEG.

    resize_mode='center_crop' (default): изображение не ресайзится, а кропается
    центральным окном img_size×img_size — интерполяция при ресайзе смещает и
    размазывает 8×8 сетку JPEG, уничтожая сам измеряемый сигнал (аудит 07.2026).
    """

    def __init__(
        self,
        img_size: int = 256,
        block_size: int = 8,
        patch_grid: int = 4,
        resize_mode: ResizeMode = "center_crop",
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.block_size = int(block_size)
        self.patch_grid = max(1, int(patch_grid))
        self.resize_mode = resize_mode

    def _blockiness_stats(self, img_f: np.ndarray) -> list[float]:
        bs = self.block_size
        _h, _w = img_f.shape
        # вертикальные разности |I(y,x) - I(y,x+1)|
        d_h = np.abs(np.diff(img_f, axis=1))
        # маска: столбец x+1 кратен bs → граница блока
        mask_h_boundary = np.zeros_like(d_h, dtype=bool)
        mask_h_boundary[:, bs - 1 :: bs] = True
        bH = d_h[mask_h_boundary].mean() if mask_h_boundary.any() else 0.0
        nH = d_h[~mask_h_boundary].mean() if (~mask_h_boundary).any() else 0.0

        d_v = np.abs(np.diff(img_f, axis=0))
        mask_v_boundary = np.zeros_like(d_v, dtype=bool)
        mask_v_boundary[bs - 1 :: bs, :] = True
        bV = d_v[mask_v_boundary].mean() if mask_v_boundary.any() else 0.0
        nV = d_v[~mask_v_boundary].mean() if (~mask_v_boundary).any() else 0.0

        # Упрощённая относительная блочность (boundary vs interior):
        blockiness_h = (bH - nH) / (nH + 1e-8)
        blockiness_v = (bV - nV) / (nV + 1e-8)
        return [
            float(bH),
            float(nH),
            float(bV),
            float(nV),
            float(blockiness_h),
            float(blockiness_v),
        ]

    def _extract(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        feats: list[float] = []
        # Патчевые статистики только при patch_grid > 1: при grid == 1 патч
        # совпадает с целым изображением и дублировал бы глобальный блок ниже.
        if self.patch_grid > 1:
            grid = self.patch_grid
            h, w = img_f.shape
            bh, bw = h // grid, w // grid
            for gy in range(grid):
                for gx in range(grid):
                    block = img_f[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                    feats.extend(self._blockiness_stats(block))
        # Глобально
        feats.extend(self._blockiness_stats(img_f))
        return np.asarray(feats, dtype=np.float32)

    #: имена статистик _blockiness_stats (порядок как в возвращаемом списке)
    _BLOCKINESS_STAT_NAMES = ("bh", "nh", "bv", "nv", "blockiness_h", "blockiness_v")

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: патчи сетки (если grid > 1), затем глобальный блок."""
        names: list[str] = []
        if self.patch_grid > 1:
            for gy in range(self.patch_grid):
                for gx in range(self.patch_grid):
                    names.extend(f"g{gy}_{gx}_{s}" for s in self._BLOCKINESS_STAT_NAMES)
        names.extend(f"global_{s}" for s in self._BLOCKINESS_STAT_NAMES)
        return names


class PRNUExtractor(_CenterCropReadMixin, BaseFeatureExtractor):
    """NOISE-3: PRNU-подобный residual.

    Полноценный PRNU требует много изображений с одной камеры. Здесь — упрощённая
    версия: denoise residual + статистики корреляции между residual-ом
    и самим изображением (PRNU предполагает мультипликативный шум, пропорциональный
    яркости). Используется как single-image proxy. Хорошо комбинируется с одним
    классом (ML-8 / One-Class SVM).

    resize_mode='center_crop' (default): изображение не ресайзится, а кропается
    центральным окном img_size×img_size — интерполяция при ресайзе усредняет
    соседние пиксели и стирает однопиксельный сенсорный шум, т.е. сам сигнал
    PRNU (аудит 07.2026).
    """

    def __init__(
        self,
        img_size: int = 256,
        denoise_method: DenoiseMethod = "nlmeans",
        denoise_strength: float = 8.0,
        patch_grid: int = 2,
        resize_mode: ResizeMode = "center_crop",
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.denoise_method = denoise_method
        self.denoise_strength = denoise_strength
        self.patch_grid = max(1, int(patch_grid))
        self.resize_mode = resize_mode

    def _extract(self, img: np.ndarray) -> np.ndarray:
        denoised = _denoise(img, self.denoise_method, self.denoise_strength)
        residual = img.astype(np.float32) - denoised.astype(np.float32)
        # PRNU должен быть пропорционален яркости — оцениваем корреляцию residual ~ I
        img_f = img.astype(np.float32)
        feats: list[float] = []
        feats.extend(_residual_stats(residual))
        # Глобальная нормированная корреляция (Pearson)
        a = residual.ravel()
        b = img_f.ravel()
        if a.std() > 1e-6 and b.std() > 1e-6:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = 0.0
        feats.append(corr)
        # Patch-wise корреляция и статистики
        if self.patch_grid > 1:
            grid = self.patch_grid
            h, w = residual.shape
            bh, bw = h // grid, w // grid
            for gy in range(grid):
                for gx in range(grid):
                    rblock = residual[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                    iblock = img_f[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                    feats.extend(_residual_stats(rblock))
                    aa, bb = rblock.ravel(), iblock.ravel()
                    if aa.std() > 1e-6 and bb.std() > 1e-6:
                        feats.append(float(np.corrcoef(aa, bb)[0, 1]))
                    else:
                        feats.append(0.0)
        return np.asarray(feats, dtype=np.float32)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: глобальные stats + corr, затем по патчам сетки."""
        names = [f"global_{s}" for s in _RESIDUAL_STAT_NAMES]
        names.append("global_corr")
        if self.patch_grid > 1:
            for gy in range(self.patch_grid):
                for gx in range(self.patch_grid):
                    p = f"g{gy}_{gx}"
                    names.extend(f"{p}_{s}" for s in _RESIDUAL_STAT_NAMES)
                    names.append(f"{p}_corr")
        return names


class NoiseResidualSRMExtractor(BaseFeatureExtractor):
    """NOISE-1: residual после денойзинга + SRM kernels → расширенные статистики.

    Параметры:
        img_size: размер ресайза.
        denoise_method: 'nlmeans' (default) | 'median' | 'bm3d'.
        denoise_strength: сила денойзера (h для NLM, ksize для median).
        use_srm: считать ли SRM-фильтры (по умолчанию True).
        use_residual: считать ли residual после денойзинга (по умолчанию True).
        patch_grid: если >1 — гистограммы статистик по сетке patch_grid×patch_grid
            (помогает RandomForest/XGBoost ловить локальную неоднородность).
    """

    def __init__(
        self,
        img_size: int = 256,
        denoise_method: DenoiseMethod = "nlmeans",
        denoise_strength: float = 10.0,
        use_srm: bool = True,
        use_residual: bool = True,
        patch_grid: int = 1,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.denoise_method = denoise_method
        self.denoise_strength = denoise_strength
        self.use_srm = use_srm
        self.use_residual = use_residual
        self.patch_grid = max(1, int(patch_grid))
        self._kernels = _srm_kernels() if use_srm else []

    def _patchwise_stats(self, residual: np.ndarray) -> list[float]:
        if self.patch_grid <= 1:
            return _residual_stats(residual)
        h, w = residual.shape
        bh, bw = h // self.patch_grid, w // self.patch_grid
        out: list[float] = []
        for gy in range(self.patch_grid):
            for gx in range(self.patch_grid):
                block = residual[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                out.extend(_residual_stats(block))
        return out

    def _extract(self, img: np.ndarray) -> np.ndarray:
        feats: list[float] = []

        # 1. Noise residual after denoising
        if self.use_residual:
            denoised = _denoise(img, self.denoise_method, self.denoise_strength)
            residual = img.astype(np.float32) - denoised.astype(np.float32)
            feats.extend(self._patchwise_stats(residual))

        # 2. SRM filtered residuals
        if self.use_srm and self._kernels:
            img_f = img.astype(np.float32)
            for kernel in self._kernels:
                resp = cv2.filter2D(img_f, ddepth=cv2.CV_32F, kernel=kernel)
                feats.extend(self._patchwise_stats(resp))

        return np.asarray(feats, dtype=np.float32)

    def _patchwise_stat_names(self, source: str) -> list[str]:
        """Имена _patchwise_stats для одной residual-карты (source — её метка)."""
        if self.patch_grid <= 1:
            return [f"{source}_{s}" for s in _RESIDUAL_STAT_NAMES]
        names: list[str] = []
        for gy in range(self.patch_grid):
            for gx in range(self.patch_grid):
                names.extend(f"{source}_g{gy}_{gx}_{s}" for s in _RESIDUAL_STAT_NAMES)
        return names

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: residual после денойзинга, затем SRM-ядра по порядку."""
        names: list[str] = []
        if self.use_residual:
            names.extend(self._patchwise_stat_names("residual"))
        if self.use_srm and self._kernels:
            for i in range(len(self._kernels)):
                names.extend(self._patchwise_stat_names(f"srm{i}"))
        return names
