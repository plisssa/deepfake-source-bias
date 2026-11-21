"""Есть ли лицо в кадре, по классам и источникам.

Зачем. При разборе дубликатов 29.09.2026 нашлась пара кадров DFF, где
центральный кроп 256×256 попал на гитару и рубашку — лица в кадре нет. DFF
приводит wiki-фото к 512×512 и подменяет ЛИЦО; если в кроп лицо не попало, в
«фейковом» кадре нет ни одного подделанного пикселя — только следы обработки
датасета. Такой кадр в принципе нельзя «детектировать как дипфейк», и если
доля таких кадров различается между классами, это ещё и канал утечки.

Детектор — каскад Хаара из OpenCV (без сети и весов): грубый, но одинаковый
для обоих классов. Важна не абсолютная доля, а РАЗНИЦА между классами.

    python -m scripts.face_presence data=dm_only +faces.n=1500
"""

from __future__ import annotations

from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def has_face(path: str, detector) -> bool | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return len(faces) > 0


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    n = int((cfg.get("faces", {}) or {}).get("n", 1500))
    split_dir = Path(cfg.paths.splits_dir) / cfg.data.name / str(cfg.data.splits_version)
    df = pd.read_csv(split_dir / "test.csv")
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rows = []
    for src, g in df.groupby("source"):
        g = g.sample(min(n, len(g)), random_state=0)
        found = [has_face(p, det) for p in g["path"]]
        ok = [f for f in found if f is not None]
        rows.append(
            {
                "source": src,
                "label": int(g["label"].iloc[0]),
                "n": len(ok),
                "face_rate": float(np.mean(ok)) if ok else float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["label", "source"])
    print(f"=== {cfg.data.name} {cfg.data.splits_version}: доля кадров с найденным лицом")
    print(out.round(3).to_string(index=False))
    by_class = out.groupby("label").apply(lambda t: np.average(t.face_rate, weights=t.n))
    print(
        f"по классам: real {by_class.get(0, float('nan')):.3f} | fake {by_class.get(1, float('nan')):.3f}"
    )
    tables = Path(cfg.paths.reports_dir) / "tables"
    out.to_csv(tables / f"face_presence_{cfg.data.name}_{cfg.data.splits_version}.csv", index=False)


if __name__ == "__main__":
    main()
