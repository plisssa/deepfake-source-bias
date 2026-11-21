"""Текстурные признаки: LBP (skimage/simple-roll), HOG, GLCM, LTP, LPQ, банк Габора."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from deepfakes.features.base import BaseFeatureExtractor

LBPMethod = Literal["uniform", "default", "ror", "var", "simple"]


def _num_token(x: float) -> str:
    """Числовой токен для имени признака: 4.0 → '4', 2.5 → '2p5', -1.5 → 'm1p5'."""
    xf = float(x)
    if xf == int(xf):
        return str(int(xf))
    return str(xf).replace(".", "p").replace("-", "m")


def _simple_roll_lbp(img: np.ndarray, radius: int = 1) -> np.ndarray:
    """Оригинальный TEX-2: LBP через np.roll по квадрату [-r, r].

    np.roll даёт циклический перенос. Решение —
    заменить на BORDER_REFLECT через cv2.copyMakeBorder и явный сдвиг.
    """
    h, w = img.shape
    pad = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    lbp = np.zeros_like(img, dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = pad[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            lbp += (shifted >= img).astype(np.uint8)
    return lbp


class HOGExtractor(BaseFeatureExtractor):
    """TEX-4: HOG через skimage.feature.hog.

    Захватывает структурные особенности изображения; полезно для выявления
    изменений формы/структуры лица.
    """

    def __init__(
        self,
        img_size: int = 128,
        orientations: int = 9,
        pixels_per_cell: int = 8,
        cells_per_block: int = 2,
        block_norm: str = "L2-Hys",
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.orientations = int(orientations)
        self.pixels_per_cell = int(pixels_per_cell)
        self.cells_per_block = int(cells_per_block)
        self.block_norm = block_norm

    def _extract(self, img: np.ndarray) -> np.ndarray:
        from skimage.feature import hog

        feats = hog(
            img,
            orientations=self.orientations,
            pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
            cells_per_block=(self.cells_per_block, self.cells_per_block),
            block_norm=self.block_norm,
            feature_vector=True,
        )
        return feats.astype(np.float32)

    def _build_feature_names(self) -> list[str]:
        """Порядок компонент HOG-вектора не интерпретируем поимённо — индексные имена.

        Размерность: (n_blocks)^2 * cells_per_block^2 * orientations; если
        transform уже прошёл — берём фактическую n_features_out_."""
        if self.n_features_out_ is not None:
            n = self.n_features_out_
        else:
            n_cells = self.img_size // self.pixels_per_cell
            n_blocks = n_cells - self.cells_per_block + 1
            n = n_blocks * n_blocks * self.cells_per_block**2 * self.orientations
        return [f"hog_{i:04d}" for i in range(n)]


class GLCMExtractor(BaseFeatureExtractor):
    """TEX-6: Haralick / GLCM (Gray-Level Co-occurrence Matrix).

    Через skimage.feature.graycomatrix + graycoprops. Признаки:
    contrast, dissimilarity, homogeneity, energy, correlation, ASM —
    по каждой паре (distance, angle), плюс энтропия GLCM.
    """

    DEFAULT_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")

    def __init__(
        self,
        img_size: int = 128,
        distances: tuple[int, ...] = (1, 2, 3),
        angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        levels: int = 32,
        patch_grid: int = 1,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.distances = tuple(distances)
        self.angles = tuple(angles)
        self.levels = int(levels)
        self.patch_grid = max(1, int(patch_grid))

    def _glcm_features(self, gray_quant: np.ndarray) -> list[float]:
        from skimage.feature import graycomatrix, graycoprops

        glcm = graycomatrix(
            gray_quant,
            distances=list(self.distances),
            angles=list(self.angles),
            levels=self.levels,
            symmetric=True,
            normed=True,
        )
        feats: list[float] = []
        for prop in self.DEFAULT_PROPS:
            vals = graycoprops(glcm, prop)
            feats.extend(vals.ravel().tolist())
        # Энтропия GLCM
        eps = 1e-12
        entropy = float(-np.sum(glcm * np.log(glcm + eps)))
        feats.append(entropy)
        return feats

    def _extract(self, img: np.ndarray) -> np.ndarray:
        # Квантование до self.levels уровней
        q = (img.astype(np.float32) / 256.0 * self.levels).astype(np.uint8)
        q = np.clip(q, 0, self.levels - 1)
        if self.patch_grid == 1:
            return np.asarray(self._glcm_features(q), dtype=np.float32)
        grid = self.patch_grid
        h, w = q.shape
        bh, bw = h // grid, w // grid
        feats: list[float] = []
        for gy in range(grid):
            for gx in range(grid):
                feats.extend(
                    self._glcm_features(q[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw])
                )
        return np.asarray(feats, dtype=np.float32)

    def _one_patch_names(self) -> list[str]:
        """Имена признаков одного вызова _glcm_features: prop × (distance × angle) + entropy.

        graycoprops возвращает (n_distances, n_angles); ravel идёт по строкам —
        внешний цикл по distance, внутренний по angle."""
        names: list[str] = []
        for prop in self.DEFAULT_PROPS:
            for d in self.distances:
                for a in self.angles:
                    names.append(f"{prop}_d{d}_a{round(float(np.rad2deg(a)))}")
        names.append("entropy")
        return names

    def _build_feature_names(self) -> list[str]:
        base = self._one_patch_names()
        if self.patch_grid == 1:
            return base
        names: list[str] = []
        for gy in range(self.patch_grid):
            for gx in range(self.patch_grid):
                names.extend(f"g{gy}_{gx}_{n}" for n in base)
        return names


class LBPExtractor(BaseFeatureExtractor):
    """TEX-2 переписанный: канонический LBP через skimage + multi-scale + baseline simple-roll.

    Режимы вычисления:
      - реализация LBP не классическая → method='uniform' (skimage) даёт честный canonical;
      - np.roll портит границы → BORDER_REFLECT (в режиме 'simple');
      - bins=256 для uniform не нужно → для uniform автоматически P+2 бина.
    """

    def __init__(
        self,
        img_size: int = 256,
        radii: tuple[int, ...] = (1, 2, 3),
        n_points: int | None = None,
        method: LBPMethod = "uniform",
        block_grid: int | None = None,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.radii = tuple(radii)
        self.n_points = n_points  # если None — 8 * r
        self.method = method
        self.block_grid = block_grid  # если задано — разбиваем на сетку block_grid x block_grid

    def _hist_for_one_radius(self, img: np.ndarray, radius: int) -> np.ndarray:
        n_pts = self.n_points or (8 * radius)
        if self.method == "simple":
            lbp = _simple_roll_lbp(img, radius=radius)
            n_bins = 256
            rng = (0, 256)
        else:
            lbp = local_binary_pattern(img, P=n_pts, R=radius, method=self.method)
            # для uniform количество бинов = P + 2
            if self.method == "uniform":
                n_bins = n_pts + 2
                rng = (0, n_bins)
            else:
                n_bins = 256
                rng = (0, 256)

        if self.block_grid:
            grid = self.block_grid
            h, w = lbp.shape
            bh, bw = h // grid, w // grid
            hists: list[np.ndarray] = []
            for gy in range(grid):
                for gx in range(grid):
                    block = lbp[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                    h_, _ = np.histogram(block.ravel(), bins=n_bins, range=rng, density=True)
                    hists.append(h_)
            return np.concatenate(hists).astype(np.float32)

        h_, _ = np.histogram(lbp.ravel(), bins=n_bins, range=rng, density=True)
        return h_.astype(np.float32)

    def _extract(self, img: np.ndarray) -> np.ndarray:
        feats = [self._hist_for_one_radius(img, r) for r in self.radii]
        return np.concatenate(feats).astype(np.float32)

    def _n_bins_for_radius(self, radius: int) -> int:
        """Число бинов гистограммы для радиуса (та же логика, что в _hist_for_one_radius)."""
        if self.method == "uniform":
            return (self.n_points or (8 * radius)) + 2
        return 256

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: радиус → (блок сетки) → бины гистограммы."""
        names: list[str] = []
        for r in self.radii:
            n_bins = self._n_bins_for_radius(r)
            pad = max(2, len(str(n_bins - 1)))
            bins = [f"bin{i:0{pad}d}" for i in range(n_bins)]
            if self.block_grid:
                for gy in range(self.block_grid):
                    for gx in range(self.block_grid):
                        names.extend(f"r{r}_g{gy}_{gx}_{b}" for b in bins)
            else:
                names.extend(f"r{r}_{b}" for b in bins)
        return names


class LTPExtractor(BaseFeatureExtractor):
    """TEX-3: Local Ternary Patterns (LTP).

    Расширение LBP с зоной нечувствительности вокруг центра: для каждого из
    8 соседей s = +1, если сосед >= центр + t; -1, если сосед <= центр - t;
    иначе 0. Тернарный паттерн раскладывается на два бинарных 8-битных кода:
    «верхний» (биты, где s == +1) и «нижний» (биты, где s == -1). Признаки —
    density-гистограммы обоих кодов на каждый радиус.

    Соседи берутся сдвигами через cv2.copyMakeBorder + BORDER_REFLECT
    (как в _simple_roll_lbp): это квадратная аппроксимация окружности
    радиуса r — 8 точек на периметре квадрата [-r, r], без билинейной
    интерполяции истинных круговых координат.
    """

    #: 8 соседей на квадрате радиуса 1 (умножаются на r); порядок задаёт биты 0..7
    _NEIGHBOR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))

    def __init__(
        self,
        img_size: int = 256,
        radii: tuple[int, ...] = (1, 2),
        t: float = 5.0,
        n_bins: int = 256,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.radii = tuple(radii)
        self.t = float(t)
        self.n_bins = int(n_bins)

    def _ltp_codes(self, img_f: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
        """Верхний и нижний 8-битные коды LTP для одного радиуса."""
        h, w = img_f.shape
        pad = cv2.copyMakeBorder(img_f, radius, radius, radius, radius, cv2.BORDER_REFLECT)
        upper = np.zeros((h, w), dtype=np.uint8)
        lower = np.zeros((h, w), dtype=np.uint8)
        for bit, (dy, dx) in enumerate(self._NEIGHBOR_OFFSETS):
            oy, ox = dy * radius, dx * radius
            nb = pad[radius + oy : radius + oy + h, radius + ox : radius + ox + w]
            upper |= (nb >= img_f + self.t).astype(np.uint8) << bit
            lower |= (nb <= img_f - self.t).astype(np.uint8) << bit
        return upper, lower

    def _extract(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        feats: list[np.ndarray] = []
        for r in self.radii:
            upper, lower = self._ltp_codes(img_f, r)
            for code in (upper, lower):
                h_, _ = np.histogram(code.ravel(), bins=self.n_bins, range=(0, 256), density=True)
                feats.append(h_.astype(np.float32))
        return np.concatenate(feats).astype(np.float32)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: радиус → (upper, lower) → бины гистограммы."""
        pad = max(2, len(str(self.n_bins - 1)))
        names: list[str] = []
        for r in self.radii:
            for kind in ("upper", "lower"):
                names.extend(f"r{r}_{kind}_bin{i:0{pad}d}" for i in range(self.n_bins))
        return names


class LPQExtractor(BaseFeatureExtractor):
    """Local Phase Quantization (LPQ): квантование локальной фазы STFT.

    В окне M x M (win_size) считаются отклики на 4 низкие частоты
    u1 = [a, 0], u2 = [0, a], u3 = [a, a], u4 = [a, -a], где a = 1/M —
    свёрткой изображения с сепарабельными комплексными ядрами exp(-2πi*u*x).
    8 бит кода — знаки Re и Im четырёх откликов -> код 0..255; признак —
    density-гистограмма из 256 бинов.

    Упрощение относительно оригинального LPQ (Ojansivu & Heikkilä, 2008):
    декорреляция откликов (whitening через модель ковариации) НЕ выполняется —
    знаки берутся напрямую с «сырых» откликов.
    """

    def __init__(self, img_size: int = 256, win_size: int = 7, verbose: bool = True):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.win_size = int(win_size)

    @staticmethod
    def _conv_sep(
        img_f: np.ndarray, ky: np.ndarray, kx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Сепарабельная комплексная свёртка: 1D по строкам (ky), затем по столбцам (kx).

        Комплексная арифметика разложена на вещественные convolve1d (float32)."""
        from scipy.ndimage import convolve1d

        tmp_r = convolve1d(img_f, ky.real.astype(np.float32), axis=0, mode="reflect")
        tmp_i = convolve1d(img_f, ky.imag.astype(np.float32), axis=0, mode="reflect")
        kxr = kx.real.astype(np.float32)
        kxi = kx.imag.astype(np.float32)
        resp_r = convolve1d(tmp_r, kxr, axis=1, mode="reflect") - convolve1d(
            tmp_i, kxi, axis=1, mode="reflect"
        )
        resp_i = convolve1d(tmp_r, kxi, axis=1, mode="reflect") + convolve1d(
            tmp_i, kxr, axis=1, mode="reflect"
        )
        return resp_r, resp_i

    def _extract(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        m = self.win_size
        a = 1.0 / m
        x = np.arange(m, dtype=np.float32) - m // 2
        w0 = np.ones(m, dtype=np.complex64)
        w1 = np.exp(-2j * np.pi * a * x).astype(np.complex64)  # exp(-2πi*a*x)
        w2 = np.conj(w1)  # exp(+2πi*a*x) — для частоты с компонентой -a
        # (ky — вертикальная компонента частоты, kx — горизонтальная)
        freq_kernels = (
            (w0, w1),  # u1 = [a, 0]
            (w1, w0),  # u2 = [0, a]
            (w1, w1),  # u3 = [a, a]
            (w1, w2),  # u4 = [a, -a]
        )
        code = np.zeros(img_f.shape, dtype=np.uint8)
        for i, (ky, kx) in enumerate(freq_kernels):
            resp_r, resp_i = self._conv_sep(img_f, ky, kx)
            code |= (resp_r >= 0).astype(np.uint8) << (2 * i)
            code |= (resp_i >= 0).astype(np.uint8) << (2 * i + 1)
        h_, _ = np.histogram(code.ravel(), bins=256, range=(0, 256), density=True)
        return h_.astype(np.float32)

    def _build_feature_names(self) -> list[str]:
        """256 бинов гистограммы LPQ-кодов."""
        return [f"lpq_bin{i:03d}" for i in range(256)]


class GaborBankExtractor(BaseFeatureExtractor):
    """TEX-5: банк фильтров Габора (cv2.getGaborKernel).

    Для каждой пары (ориентация θ, длина волны λ) строится ядро с
    sigma = 0.56 * λ, gamma = 0.5, psi = 0; отклик — cv2.filter2D (float32).
    Признаки на пару: mean(|r|), std(r), energy = mean(r^2). При дефолтных
    4 ориентациях и 3 длинах волн — 36 признаков; patch_grid > 1 считает
    те же статистики по каждому патчу сетки (как в GLCMExtractor).
    """

    def __init__(
        self,
        img_size: int = 128,
        orientations: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
        wavelengths: tuple[float, ...] = (4.0, 8.0, 16.0),
        gamma: float = 0.5,
        psi: float = 0.0,
        patch_grid: int = 1,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.orientations = tuple(orientations)  # в градусах
        self.wavelengths = tuple(wavelengths)
        self.gamma = float(gamma)
        self.psi = float(psi)
        self.patch_grid = max(1, int(patch_grid))

    #: имена статистик отклика (порядок как в _response_stats)
    _STAT_NAMES = ("mean", "std", "energy")

    @staticmethod
    def _response_stats(resp: np.ndarray) -> list[float]:
        return [
            float(np.mean(np.abs(resp))),
            float(np.std(resp)),
            float(np.mean(resp**2)),  # energy
        ]

    def _extract(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32) / 255.0
        feats: list[float] = []
        grid = self.patch_grid
        h, w = img_f.shape
        bh, bw = h // grid, w // grid
        for theta_deg in self.orientations:
            for lam in self.wavelengths:
                sigma = 0.56 * lam
                ksize = int(2 * np.ceil(3.0 * sigma) + 1)
                kernel = cv2.getGaborKernel(
                    (ksize, ksize),
                    sigma,
                    np.deg2rad(theta_deg),
                    lam,
                    self.gamma,
                    self.psi,
                    ktype=cv2.CV_32F,
                )
                resp = cv2.filter2D(img_f, cv2.CV_32F, kernel)
                if grid == 1:
                    feats.extend(self._response_stats(resp))
                    continue
                for gy in range(grid):
                    for gx in range(grid):
                        block = resp[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
                        feats.extend(self._response_stats(block))
        return np.asarray(feats, dtype=np.float32)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: θ → λ → (блок сетки) → mean/std/energy."""
        names: list[str] = []
        grid = self.patch_grid
        for theta_deg in self.orientations:
            for lam in self.wavelengths:
                prefix = f"theta{_num_token(theta_deg)}_lam{_num_token(lam)}"
                if grid == 1:
                    names.extend(f"{prefix}_{s}" for s in self._STAT_NAMES)
                    continue
                for gy in range(grid):
                    for gx in range(grid):
                        names.extend(f"{prefix}_g{gy}_{gx}_{s}" for s in self._STAT_NAMES)
        return names
