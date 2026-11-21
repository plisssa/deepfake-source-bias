"""Частотные признаки: FFT bins + spectrum slope + flatness, FREQ-1/3/4/5."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import stats

from deepfakes.features.base import BaseFeatureExtractor
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


# ---------- общие спектральные помощники (FREQ-1 и FREQ-4) ----------


def _hann2d(h: int, w: int) -> np.ndarray:
    """Сепарабельное 2D-окно Ханна — против spectral leakage от краёв кадра."""
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _windowed_fft(img: np.ndarray, window: str) -> np.ndarray:
    """FFT изображения с опциональным окном; возвращает fftshift-нутый спектр.

    Среднее вычитается ДО окна: иначе огромная DC-компонента (яркость ~128)
    протекает через боковые лепестки окна на малые радиусы и портит slope
    (у белого шума он получался около -1 вместо ~0)."""
    x = img.astype(np.float32)
    x = x - x.mean()
    if window == "hann":
        x = x * _hann2d(*x.shape)
    elif window != "none":
        raise ValueError(f"window должен быть 'hann' или 'none', получено: {window!r}")
    return np.fft.fftshift(np.fft.fft2(x))


def _spectral_grids(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сетки для fftshift-нутого спектра: нормированный радиус (0..sqrt(2)),
    радиус в пикселях и угол по модулю pi (спектр центросимметричен)."""
    h, w = shape
    cy, cx = h // 2, w // 2
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    X, Y = np.meshgrid(x, y)
    r_norm = np.sqrt((X / (w / 2)) ** 2 + (Y / (h / 2)) ** 2)
    r_pix = np.sqrt(X.astype(np.float64) ** 2 + Y.astype(np.float64) ** 2)
    theta = np.mod(np.arctan2(Y, X), np.pi)
    return r_norm, r_pix, theta


def _azimuthal_profile(ps: np.ndarray, r_pix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Азимутально усреднённый 1D-профиль мощности (Durall et al.).

    Целочисленный радиус r = round(r_pix); на каждый r из [1, r_max] — средняя
    мощность по кольцу. DC (r=0) исключён. r_max = min(h, w) // 2 (Найквист по
    короткой оси; углы спектра в профиль не входят — там кольца неполные)."""
    r_int = np.round(r_pix).astype(np.int64).ravel()
    r_max = min(ps.shape) // 2
    sums = np.bincount(r_int, weights=ps.ravel().astype(np.float64))
    counts = np.bincount(r_int)
    n = min(len(sums), r_max + 1)
    prof = sums[1:n] / np.maximum(counts[1:n], 1)
    return np.arange(1, n, dtype=np.float64), prof


def _spectral_slope(ps: np.ndarray, r_pix: np.ndarray) -> float:
    """Наклон log-log линрегрессии по азимутальному профилю (НЕ по 2D-пикселям)."""
    radii, prof = _azimuthal_profile(ps, r_pix)
    ok = prof > 1e-12
    if ok.sum() < 3:
        return 0.0
    slope, _, _, _, _ = stats.linregress(np.log(radii[ok]), np.log(prof[ok]))
    return float(slope)


def _spectral_flatness(ps: np.ndarray, dc_mask: np.ndarray) -> float:
    """geometric/arithmetic mean мощности БЕЗ DC-компоненты; в (0, 1]."""
    vals = ps[~dc_mask]
    vals = vals[vals > 1e-12]
    if vals.size == 0:
        return 0.0
    gm = np.exp(np.mean(np.log(vals)))
    am = float(vals.mean())
    return float(gm / (am + 1e-12))


def _ring_masks(r_norm: np.ndarray, dc_mask: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Кольцевые маски: edges = linspace(0, sqrt(2), n_bins+1), DC исключён.

    Последний бин включает верхнюю границу (углы спектра, r = sqrt(2))."""
    edges = np.linspace(0.0, np.sqrt(2.0), n_bins + 1)
    masks: list[np.ndarray] = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        upper = (r_norm <= hi) if i == n_bins - 1 else (r_norm < hi)
        masks.append((r_norm >= lo) & upper & ~dc_mask)
    return masks


class HFEnergyExtractor(BaseFeatureExtractor):
    """FREQ-4 переписанный: больше бинов, spectrum slope, flatness, доп. статистики.

    Решение: расширить до N кольцевых бинов + spectrum slope + flatness + mean/std.

    Мат.фиксы (аудит июля 2026, дефолтное поведение изменено осознанно):
      * slope — по азимутально усреднённому 1D-профилю (Durall et al.),
        а не по всем пикселям 2D-спектра;
      * flatness — без DC-компоненты;
      * окно Ханна перед FFT (window='hann') против spectral leakage;
      * кольцевые бины покрывают спектр целиком (edges до sqrt(2), раньше 1.5 —
        последний бин был пуст), DC исключён, энергии нормируются на суммарную
        энергию спектра (normalize=True).
    """

    def __init__(
        self,
        img_size: int = 256,
        n_bins: int = 8,
        include_slope: bool = True,
        include_flatness: bool = True,
        include_basic_stats: bool = True,
        window: str = "hann",
        normalize: bool = True,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.n_bins = n_bins
        self.include_slope = include_slope
        self.include_flatness = include_flatness
        self.include_basic_stats = include_basic_stats
        self.window = window
        self.normalize = normalize

    def _extract(self, img: np.ndarray) -> np.ndarray:
        f = _windowed_fft(img, self.window)
        ps = np.abs(f) ** 2
        r_norm, r_pix, _ = _spectral_grids(ps.shape)
        dc = r_pix == 0

        total = float(ps[~dc].sum())
        feats: list[float] = []
        for mask in _ring_masks(r_norm, dc, self.n_bins):
            e = float(ps[mask].sum())
            feats.append(e / (total + 1e-12) if self.normalize else float(np.log1p(e)))

        if self.include_slope:
            feats.append(_spectral_slope(ps, r_pix))

        if self.include_flatness:
            feats.append(_spectral_flatness(ps, dc))

        if self.include_basic_stats:
            feats.append(float(img.mean()))
            feats.append(float(img.std()))

        return np.asarray(feats, dtype=np.float32)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке конкатенации _extract: кольца → slope → flatness → mean/std."""
        names = [f"ring_{i:02d}" for i in range(self.n_bins)]
        if self.include_slope:
            names.append("slope")
        if self.include_flatness:
            names.append("flatness")
        if self.include_basic_stats:
            names.extend(["mean", "std"])
        return names


class FullFFTExtractor(BaseFeatureExtractor):
    """FREQ-1: FFT всего изображения (grayscale, окно Ханна) → спектральный дескриптор.

    Признаки:
      * азимутальный лог-профиль мощности: radial_bins значений log1p(средней
        мощности по кольцу);
      * угловой профиль: доля энергии в angular_bins секторах (углы по модулю
        pi — спектр вещественного сигнала центросимметричен), DC исключён;
      * spectrum slope + flatness (те же фиксленные формулы, что в FREQ-4);
      * опц. include_phase: mean resultant length фазы по кольцам (radial_bins
        значений) — мера когерентности фазы: у периодических GAN-артефактов
        фаза по кольцу когерентнее, чем у естественного контента.

    Итого radial_bins + angular_bins + 2 (+ radial_bins при include_phase).
    """

    def __init__(
        self,
        img_size: int = 256,
        radial_bins: int = 16,
        angular_bins: int = 8,
        window: str = "hann",
        include_phase: bool = True,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self.window = window
        self.include_phase = include_phase

    def _extract(self, img: np.ndarray) -> np.ndarray:
        f = _windowed_fft(img, self.window)
        ps = np.abs(f) ** 2
        r_norm, r_pix, theta = _spectral_grids(ps.shape)
        dc = r_pix == 0
        rings = _ring_masks(r_norm, dc, self.radial_bins)

        feats: list[float] = []
        # 1) азимутальный лог-профиль мощности
        for mask in rings:
            feats.append(float(np.log1p(ps[mask].mean())) if mask.any() else 0.0)

        # 2) угловой профиль энергии по секторам
        total = float(ps[~dc].sum())
        a_edges = np.linspace(0.0, np.pi, self.angular_bins + 1)
        for lo, hi in zip(a_edges[:-1], a_edges[1:]):
            mask = (theta >= lo) & (theta < hi) & ~dc
            feats.append(float(ps[mask].sum()) / (total + 1e-12))

        # 3) slope + flatness
        feats.append(_spectral_slope(ps, r_pix))
        feats.append(_spectral_flatness(ps, dc))

        # 4) когерентность фазы по кольцам
        if self.include_phase:
            phase = np.angle(f)
            for mask in rings:
                if mask.any():
                    feats.append(float(np.abs(np.exp(1j * phase[mask]).mean())))
                else:
                    feats.append(0.0)

        return np.asarray(feats, dtype=np.float32)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке _extract: radial-профиль → angular-сектора → slope/flatness → фаза."""
        names = [f"radial_{i:02d}" for i in range(self.radial_bins)]
        names += [f"angular_{i:02d}" for i in range(self.angular_bins)]
        names += ["slope", "flatness"]
        if self.include_phase:
            names += [f"phase_{i:02d}" for i in range(self.radial_bins)]
        return names


class CWTScalogramExtractor(BaseFeatureExtractor):
    """FREQ-5: 1D CWT (Morlet / Mexican Hat) над проекциями изображения.

    Используются вейвлеты Morlet и Mexican Hat. 2D CWT в PyWavelets нет, поэтому используется проекционный
    подход: изображение сворачивается в два 1D-сигнала — средние по строкам и
    по столбцам (каждый длины img_size, центрирован вычитанием среднего), к
    каждому применяется pywt.cwt с масштабами scales. На пару (сигнал, scale)
    берутся mean(|c|), std(c), energy(c) => 2 * len(scales) * 3 признаков
    (при дефолтных scales=(2, 4, 8, 16) — 24).

    Ограничение проекционного подхода: паттерны, сбалансированные вдоль строк
    и столбцов (например, диагональная шахматка), в проекциях взаимно
    сокращаются и этим экстрактором не видны; целевые артефакты —
    периодический banding и низкочастотная структура.
    """

    def __init__(
        self,
        img_size: int = 256,
        wavelet: str = "morl",
        scales: tuple[int, ...] = (2, 4, 8, 16),
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.wavelet = wavelet
        self.scales = tuple(int(s) for s in scales)
        try:
            import pywt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PyWavelets обязателен для FREQ-5 (CWTScalogramExtractor): "
                "pip install PyWavelets"
            ) from e

    def _extract(self, img: np.ndarray) -> np.ndarray:
        import pywt

        img_f = img.astype(np.float32)
        feats: list[float] = []
        for sig in (img_f.mean(axis=1), img_f.mean(axis=0)):
            sig = sig - sig.mean()
            coeffs, _ = pywt.cwt(sig, self.scales, self.wavelet)
            for c in coeffs:  # (len(sig),) на каждый scale
                feats.append(float(np.abs(c).mean()))
                feats.append(float(c.std()))
                feats.append(float((c**2).mean()))
        return np.asarray(feats, dtype=np.float32)


class DCTBlockExtractor(BaseFeatureExtractor):
    """FREQ-2: блочный DCT 8×8 или 16×16 → статистики по AC/DC коэффициентам.

    На каждый блок: DC + энергия AC + std/kurtosis + энергия в low/mid/high
    подобластях зигзаг-порядка. Затем аггрегация по всем блокам:
    mean/std/percentiles.
    """

    #: имена 7 поблочных признаков (порядок как в _block_features)
    _BLOCK_FEATURE_NAMES = ("dc", "ac_energy", "ac_std", "ac_kurt", "low", "mid", "high")

    def __init__(
        self,
        img_size: int = 256,
        block_size: int = 8,
        log_scale: bool = True,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.block_size = int(block_size)
        self.log_scale = log_scale

    def _block_features(self, block: np.ndarray) -> np.ndarray:
        # DCT блока: cv2.dct требует float32 и 2D
        d = cv2.dct(block.astype(np.float32))
        bs = self.block_size
        if self.log_scale:
            d = np.log1p(np.abs(d)) * np.sign(d)
        # DC и AC
        dc = float(d[0, 0])
        ac = d.copy()
        ac[0, 0] = 0.0
        ac_energy = float(np.sum(ac * ac))
        ac_std = float(ac.std())
        ac_kurt = float(((ac - ac.mean()) ** 4).mean() / (ac.std() ** 4 + 1e-12))
        # low/mid/high по зигзаг-индексам
        ii, jj = np.indices((bs, bs))
        z = ii + jj
        low = float((d[z <= bs // 3] ** 2).sum())
        mid = float((d[(z > bs // 3) & (z <= 2 * bs // 3)] ** 2).sum())
        high = float((d[z > 2 * bs // 3] ** 2).sum())
        return np.array([dc, ac_energy, ac_std, ac_kurt, low, mid, high], dtype=np.float32)

    def _extract(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32) - 128.0  # центрировать, как в JPEG
        bs = self.block_size
        h, w = img_f.shape
        block_feats: list[np.ndarray] = []
        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block_feats.append(self._block_features(img_f[y : y + bs, x : x + bs]))
        if not block_feats:
            return np.zeros(7 * 5, dtype=np.float32)
        bf = np.vstack(block_feats)  # (n_blocks, 7)
        # агрегаты на блок-уровне: mean, std, p10, p50, p90
        agg = np.concatenate(
            [
                bf.mean(axis=0),
                bf.std(axis=0),
                np.percentile(bf, 10, axis=0),
                np.percentile(bf, 50, axis=0),
                np.percentile(bf, 90, axis=0),
            ]
        ).astype(np.float32)
        return agg

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке np.concatenate агрегатов: mean → std → p10 → p50 → p90 (по 7)."""
        aggs = ("mean", "std", "p10", "p50", "p90")
        return [f"{agg}_{feat}" for agg in aggs for feat in self._BLOCK_FEATURE_NAMES]


class WaveletExtractor(BaseFeatureExtractor):
    """FREQ-3: multi-level DWT → статистики по subbands LL/LH/HL/HH.

    Требует `PyWavelets` (обязательная зависимость проекта). Молчаливый
    fallback на Laplacian-пирамиду удалён: он незаметно менял и семантику,
    и размерность признаков в зависимости от окружения.
    """

    def __init__(
        self,
        img_size: int = 256,
        wavelet: str = "db2",
        level: int = 3,
        verbose: bool = True,
    ):
        super().__init__(img_size=img_size, grayscale=True, verbose=verbose)
        self.wavelet = wavelet
        self.level = int(level)
        try:
            import pywt  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PyWavelets обязателен для FREQ-3 (WaveletExtractor): " "pip install PyWavelets"
            ) from e

    #: имена статистик одного subband-а (порядок как в _subband_stats)
    _SUBBAND_STAT_NAMES = ("mean", "std", "abs_mean", "kurt", "energy")

    @staticmethod
    def _subband_stats(arr: np.ndarray) -> list[float]:
        v = arr.reshape(-1).astype(np.float64)
        if v.std() < 1e-8:
            return [float(v.mean()), 0.0, 0.0, 0.0, 0.0]
        return [
            float(v.mean()),
            float(v.std()),
            float(np.abs(v).mean()),
            float(((v - v.mean()) ** 4).mean() / (v.std() ** 4 + 1e-12)),
            float((v**2).mean()),
        ]

    def _wavelet_features(self, img: np.ndarray) -> np.ndarray:
        import pywt

        coeffs = pywt.wavedec2(img.astype(np.float32), self.wavelet, level=self.level)
        feats: list[float] = []
        # coeffs = [LL_n, (LH_n, HL_n, HH_n), (LH_{n-1}, ...), ...]
        ll = coeffs[0]
        feats.extend(self._subband_stats(ll))
        for detail_level in coeffs[1:]:
            lh, hl, hh = detail_level
            feats.extend(self._subband_stats(lh))
            feats.extend(self._subband_stats(hl))
            feats.extend(self._subband_stats(hh))
        return np.asarray(feats, dtype=np.float32)

    def _extract(self, img: np.ndarray) -> np.ndarray:
        return self._wavelet_features(img)

    def _build_feature_names(self) -> list[str]:
        """Имена в порядке wavedec2: LL_n, затем (LH, HL, HH) от уровня n к уровню 1."""
        names = [f"LL{self.level}_{s}" for s in self._SUBBAND_STAT_NAMES]
        for i in range(self.level):
            lvl = self.level - i
            for band in ("LH", "HL", "HH"):
                names.extend(f"{band}{lvl}_{s}" for s in self._SUBBAND_STAT_NAMES)
        return names


def hf_gan_augment(img: np.ndarray, intensity: float = 0.02, threshold: float = 0.35) -> np.ndarray:
    """High-frequency GAN-like аугментация (использовалась в FREQ-4)."""
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X / (w / 2)) ** 2 + (Y / (h / 2)) ** 2)
    mask = (threshold < R).astype(np.float32)
    fshift_aug = fshift * (1 + intensity * mask)
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(fshift_aug)))
    return np.clip(img_back, 0, 255).astype(np.uint8)
