"""INT-5: LIME per-image объяснения для произвольного pipeline.predict_proba.

Алгоритм: LimeImageExplainer перебирает суперпиксели изображения (через quickshift),
выключает их случайными подмножествами, считает изменение вероятности и подгоняет
локальную линейную модель. На выходе — карта «влияющих» суперпикселей.

Чтобы LIME мог дёргать наш классификатор, оборачиваем pipeline.predict_proba
в функцию, которая принимает np.ndarray батча картинок (uint8 RGB) и возвращает
вероятности обоих классов (n, 2). Для этого временно сохраняем картинки на диск
(LIME выдаёт массивы, а extractor-ы хотят пути).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


def _predict_fn_from_pipeline(pipeline) -> Callable[[np.ndarray], np.ndarray]:
    """Обернуть pipeline в callable(batch_of_RGB_uint8_imgs) → (n, 2) probabilities."""

    def predict_batch(batch: np.ndarray) -> np.ndarray:
        # batch: (n, H, W, 3), uint8 RGB
        # TemporaryDirectory с автоочисткой: mkdtemp без удаления накапливал
        # тысячи PNG за один explain_topk (num_samples × картинки).
        with tempfile.TemporaryDirectory(prefix="lime_batch_") as tmp:
            tmpdir = Path(tmp)
            paths = []
            for i, img in enumerate(batch):
                p = tmpdir / f"lime_{i}.png"
                Image.fromarray(img).save(p)
                paths.append(str(p))
            proba_pos = pipeline.predict_proba(np.array(paths, dtype=object))
        # Возвращаем (n, 2): [P(real), P(fake)]
        proba_pos = np.clip(proba_pos, 1e-6, 1 - 1e-6)
        return np.stack([1.0 - proba_pos, proba_pos], axis=1)

    return predict_batch


def explain_one(
    pipeline,
    image_path: str,
    out_path: str | Path,
    img_size: int = 256,
    num_samples: int = 500,
    num_features: int = 5,
    positive_only: bool = True,
    title: str | None = None,
) -> Path | None:
    """LIME-объяснение для одной картинки. Сохраняет PNG (оригинал + маска)."""
    try:
        from lime.lime_image import LimeImageExplainer
        from skimage.segmentation import mark_boundaries
    except ImportError:
        log.warning("Install 'lime' and 'scikit-image' to use INT-5 LIME explanations")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        log.warning(f"Can't read {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    explainer = LimeImageExplainer()
    predict_fn = _predict_fn_from_pipeline(pipeline)
    exp = explainer.explain_instance(
        img,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        random_seed=42,
    )

    # Label 1 = fake (тот, который пайплайн считает позитивным)
    label = 1
    temp, mask = exp.get_image_and_mask(
        label,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False,
    )
    overlay = mark_boundaries(temp / 255.0, mask)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(title or f"LIME (label=1, top-{num_features})")
    axes[1].axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def explain_topk(
    pipeline,
    image_paths: np.ndarray,
    y_true: np.ndarray,
    out_dir: str | Path,
    k: int = 5,
    img_size: int = 256,
    num_samples: int = 500,
) -> list[Path]:
    """Объяснить топ-K по уверенности fake и топ-K реальных, ошибочно классифицированных."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pipeline.predict_proba(image_paths)

    # Top fake-confident (label = 1)
    fake_idx = np.argsort(scores)[::-1][:k]
    # Top real-but-flagged-fake (FP); обрезаем по фактическому числу FP —
    # иначе при FP < k в «top_FP» добирались бы не-FP примеры
    fp_mask = (y_true == 0) & (scores >= 0.5)
    n_fp = int(fp_mask.sum())
    fp_idx = np.argsort(np.where(fp_mask, scores, -np.inf))[::-1][: min(k, n_fp)]

    results: list[Path] = []
    for tag, idx in [("top_fake", fake_idx), ("top_FP", fp_idx)]:
        for rank, i in enumerate(idx):
            out_path = out_dir / f"lime_{tag}_{rank:02d}__score{scores[i]:.3f}.png"
            p = explain_one(
                pipeline,
                str(image_paths[i]),
                out_path,
                img_size=img_size,
                num_samples=num_samples,
                title=f"{tag} #{rank} | score={scores[i]:.3f} | y_true={int(y_true[i])}",
            )
            if p is not None:
                results.append(p)
    return results
