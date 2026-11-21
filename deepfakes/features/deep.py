"""Extract embeddings using pretrained vision backbones with explicit input normalization."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from tqdm.auto import tqdm

from deepfakes.features.base import BaseFeatureExtractor
from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def center_crop_or_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Центральный кроп size×size БЕЗ интерполяции.

    Для детекции подделок это важно: ресайз (даже INTER_AREA 256→224) гасит
    высокие частоты, в которых живут артефакты генерации. UniversalFakeDetect
    (Ojha et al., CVPR 2023) по той же причине подаёт в CLIP центральный кроп.
    Если сторона меньше size, изображение дополняется отражением — тоже без
    интерполяции (у наших нормализованных данных 256 ≥ 224, дополнение не нужно)."""
    h, w = img.shape[:2]
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        pad = [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)] + [(0, 0)] * (img.ndim - 2)
        img = np.pad(img, pad, mode="reflect")
        h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top : top + size, left : left + size]


def iter_images_ordered(paths, read_fn, n_threads: int = 8, prefetch: int = 256):
    """(индекс, изображение или None) в ИСХОДНОМ порядке, чтение — пулом потоков.

    Зачем: 29.09.2026 извлечение CLIP шло 23.8 изображения/с на V100 — видеокарта
    ждала однопоточное чтение с общей ФС. cv2.imread/imdecode отпускают GIL, так
    что потоки параллельны по-настоящему. Предвыборка ограничена `prefetch`
    кадрами: при медленной модели в памяти не копятся все изображения выборки.
    Порядок выдачи строго исходный — иначе признаки перепутались бы с путями.
    Нечитаемый файл даёт None (и предупреждение), а не исключение."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def safe(i):
        try:
            return read_fn(paths[i])
        except Exception as e:
            log.warning(f"  [extract failed] {paths[i]}: {e}")
            return None

    n = len(paths)
    with ThreadPoolExecutor(max_workers=max(1, int(n_threads))) as ex:
        inflight: deque = deque()
        nxt = 0
        while nxt < n and len(inflight) < prefetch:
            inflight.append((nxt, ex.submit(safe, nxt)))
            nxt += 1
        while inflight:
            i, fut = inflight.popleft()
            if nxt < n:
                inflight.append((nxt, ex.submit(safe, nxt)))
                nxt += 1
            yield i, fut.result()


class DeepFeatureExtractor(BaseFeatureExtractor):
    """Замороженный timm-бэкбон → вектор эмбеддинга (num_classes=0, global pool).

    Извлечение батчевое (transform переопределён): joblib-параллелизм по
    изображениям здесь вреден (модель одна, форвард и так векторизован),
    поэтому n_jobs принудительно 1.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        img_size: int = 224,
        pretrained: bool = True,
        batch_size: int = 32,
        device: str = "auto",
        verbose: bool = True,
        resize_mode: str = "resize",
        normalize: str = "model",
        model_img_size: int | None = None,
        amp: bool = True,
    ):
        """
        resize_mode: 'resize' (INTER_AREA, прежнее поведение) или 'center_crop'
            (без интерполяции — сохраняет высокие частоты, как у Ojha et al.).
        normalize: 'model' — mean/std из конфигурации КОНКРЕТНОГО чекпойнта timm
            (у OpenAI CLIP и у AugReg-ViT они не ImageNet); 'imagenet' — прежнее
            жёстко зашитое поведение. До 29.09.2026 была только ImageNet-
            нормализация, поэтому CLIP был бы искусственно ослаблен, а ViT-tiny
            (AugReg, mean=std=0.5) в раннем гибриде получал неверный вход.
        amp: на GPU считать в float16 (autocast) — в 2–3 раза быстрее для ViT-L;
            выход приводится к float32. Меняет признаки в младших знаках, поэтому
            входит в ключ кэша (get_params).
        model_img_size: передаётся в timm.create_model(img_size=...) — нужно
            моделям с нативным входом не 224 (DINOv2: 518); позиционные
            эмбеддинги интерполируются timm при загрузке.
        """
        super().__init__(img_size=img_size, grayscale=False, verbose=verbose)
        if resize_mode not in ("resize", "center_crop"):
            raise ValueError(f"resize_mode: 'resize'|'center_crop', получено {resize_mode!r}")
        if normalize not in ("model", "imagenet"):
            raise ValueError(f"normalize: 'model'|'imagenet', получено {normalize!r}")
        self.backbone = backbone
        self.pretrained = pretrained
        self.batch_size = int(batch_size)
        self.device = device
        self.resize_mode = resize_mode
        self.normalize = normalize
        self.model_img_size = model_img_size
        self.amp = bool(amp)
        self._model = None
        self._torch = None
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self._std = np.array(IMAGENET_STD, dtype=np.float32)
        self.n_jobs = 1  # параллелим батчами внутри модели, не процессами

    def __getstate__(self):
        """Сериализация без модуля torch и без весов бэкбона.

        29.09.2026 сохранение пайплайна падало: `cannot pickle 'module' object`
        (атрибут `_torch`). Замороженный предобученный бэкбон полностью задаётся
        своими параметрами и восстанавливается из кэша timm при первом
        извлечении (`_ensure_model`); класть его в каждый `pipeline.joblib` —
        это лишние 1.2 ГБ на файл у CLIP ViT-L/14 и прямой путь в квоту диска."""
        state = self.__dict__.copy()
        state["_model"] = None
        state["_torch"] = None
        return state

    #: как вёл себя экстрактор ДО 29.09.2026 — для объектов, сохранённых старым
    #: кодом. Важно восстановить именно старое поведение, а не новые умолчания:
    #: модель обучалась на ImageNet-нормализации, ресайзе и fp32, и другой вход
    #: молча дал бы другие предсказания.
    _LEGACY_STATE: ClassVar[dict[str, object]] = {
        "resize_mode": "resize",
        "normalize": "imagenet",
        "model_img_size": None,
        "amp": False,
    }

    def __setstate__(self, state):
        """Загрузка объектов, сохранённых до появления новых полей.

        29.09.2026 пересчёт статистики упал на историческом ViT-гибриде (сохранён
        27.09): у распакованного экстрактора не было resize_mode/normalize/amp,
        и извлечение падало на каждом изображении."""
        for k, v in self._LEGACY_STATE.items():
            state.setdefault(k, v)
        state.setdefault("_mean", np.array(IMAGENET_MEAN, dtype=np.float32))
        state.setdefault("_std", np.array(IMAGENET_STD, dtype=np.float32))
        self.__dict__.update(state)

    def _read_image(self, path: str) -> np.ndarray:
        if self.resize_mode == "resize":
            return super()._read_image(path)
        import cv2

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Can't read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return center_crop_or_pad(img, self.img_size)

    def _prepare_raw(self, img_bgr: np.ndarray) -> np.ndarray:
        """Тот же вход, что и в _read_image, но из уже прочитанного кадра —
        путь HybridExtractor. Без этого переопределения кроп работал бы только
        в одном из двух путей чтения."""
        if self.resize_mode == "resize":
            return super()._prepare_raw(img_bgr)
        import cv2

        return center_crop_or_pad(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), self.img_size)

    # ---------- lazy init ----------

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import timm
            import torch
        except ImportError as e:  # pragma: no cover - тестируется скипом
            raise ImportError(
                "DeepFeatureExtractor требует torch и timm: pip install -e '.[dl]'"
            ) from e
        self._torch = torch
        dev = self.device
        if dev == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        self._resolved_device = dev
        kw = {"img_size": int(self.model_img_size)} if self.model_img_size else {}
        self._model = timm.create_model(
            self.backbone, pretrained=self.pretrained, num_classes=0, **kw
        )
        self._model.eval().to(dev)
        for p in self._model.parameters():
            p.requires_grad_(False)
        if self.normalize == "model":
            cfg = timm.data.resolve_model_data_config(self._model)
            self._mean = np.array(cfg["mean"], dtype=np.float32)
            self._std = np.array(cfg["std"], dtype=np.float32)
        log.info(
            f"[deep] frozen backbone={self.backbone} device={dev} "
            f"input={self.resize_mode}@{self.img_size} "
            f"mean={np.round(self._mean, 4).tolist()} std={np.round(self._std, 4).tolist()}"
        )

    # ---------- extraction ----------

    def _to_tensor_batch(self, imgs: list[np.ndarray]):
        arr = np.stack(imgs).astype(np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        t = self._torch.from_numpy(arr.transpose(0, 3, 1, 2))
        return t.to(self._resolved_device)

    def _extract(self, img: np.ndarray) -> np.ndarray:
        # используется HybridExtractor-ом (по одному изображению)
        self._ensure_model()
        with self._torch.no_grad():
            emb = self._model(self._to_tensor_batch([img]))
        return emb[0].cpu().numpy().astype(np.float32)

    def transform(self, X) -> np.ndarray:
        import os

        self._ensure_model()
        torch = self._torch
        paths = [str(p) for p in X]
        feats: list[np.ndarray | None] = [None] * len(paths)
        batch_imgs: list[np.ndarray] = []
        batch_idx: list[int] = []
        use_amp = self.amp and self._resolved_device == "cuda"

        def _flush():
            if not batch_imgs:
                return
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp),
            ):
                emb = self._model(self._to_tensor_batch(batch_imgs)).float().cpu().numpy()
            for j, bi in enumerate(batch_idx):
                feats[bi] = emb[j].astype(np.float32)
            batch_imgs.clear()
            batch_idx.clear()

        n_threads = int(os.environ.get("DEEPFAKES_N_JOBS", "8"))
        bar = tqdm(total=len(paths), desc="DeepFeatureExtractor", disable=not self.verbose)
        for i, img in iter_images_ordered(
            paths, self._read_image, n_threads=n_threads, prefetch=4 * self.batch_size
        ):
            bar.update(1)
            if img is not None:
                batch_imgs.append(img)
                batch_idx.append(i)
            if len(batch_imgs) >= self.batch_size:
                _flush()
        _flush()
        bar.close()
        X_out = self._finalize(feats)
        self._ensure_feature_names()
        return X_out

    def _build_feature_names(self) -> list[str] | None:
        if self.n_features_out_ is None:
            return None
        return [f"deep_{self.backbone}_{i:04d}" for i in range(self.n_features_out_)]
