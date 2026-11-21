"""Аудит датасета: размеры, форматы, EXIF, perceptual-дубли, базовая статистика яркости.

Запуск (audit_target — новый ключ, поэтому с «+»; формат kind:name):
    python -m scripts.audit_dataset data=default +audit_target=real:archive_train
    python -m scripts.audit_dataset data=default +audit_target=fake:redface

Без +audit_target аудируется первый real-источник.
Сохраняет CSV отчёт в data/CARDS/<source>_audit.csv и markdown-сводку.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from deepfakes.io.collect import collect_images
from deepfakes.utils.logging import configure_logging, get_logger


def _audit_one(p: str) -> dict:
    out = {"path": p, "ok": False}
    try:
        with Image.open(p) as img:
            out["width"], out["height"] = img.size
            out["format"] = img.format or ""
            out["mode"] = img.mode or ""
            exif = getattr(img, "_getexif", lambda: None)() or {}
            out["has_exif"] = bool(exif)
        arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            out["mean_lum"] = float(arr.mean())
            out["std_lum"] = float(arr.std())
        out["bytes"] = Path(p).stat().st_size
        out["ok"] = True
    except Exception as e:
        out["error"] = str(e)
    return out


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)

    target = cfg.get("audit_target")  # ожидаем формат "real:archive_train" или "fake:redface"
    if target is None:
        target = "real:" + cfg.data.real_sources[0].name
    kind, name = target.split(":")
    sources = cfg.data.real_sources if kind == "real" else cfg.data.fake_sources
    src = next(s for s in sources if s.name == name)
    path = OmegaConf.to_container(src, resolve=True)["path"]

    log.info(f"Auditing {kind}:{name} at {path}")
    files = collect_images(path, verify=False)
    log.info(f"Found {len(files)} files")

    rows = [_audit_one(p) for p in tqdm(files, desc=f"audit {name}")]
    df = pd.DataFrame(rows)

    out_dir = Path("data/CARDS")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{kind}_{name}_audit.csv"
    df.to_csv(csv_path, index=False)

    # Краткая сводка
    ok = df[df["ok"]]
    summary = {
        "n_files_total": len(df),
        "n_files_ok": len(ok),
        "n_files_bad": int(len(df) - len(ok)),
        "format_counts": Counter(ok["format"]).most_common(),
        "mode_counts": Counter(ok["mode"]).most_common(),
        "width_quantiles": np.quantile(ok["width"], [0, 0.5, 1]).tolist() if len(ok) else [],
        "height_quantiles": np.quantile(ok["height"], [0, 0.5, 1]).tolist() if len(ok) else [],
        "mean_lum_quantiles": np.quantile(ok["mean_lum"], [0, 0.5, 1]).tolist() if len(ok) else [],
        "median_bytes": float(np.median(ok["bytes"])) if len(ok) else 0.0,
        "has_exif_fraction": float(ok["has_exif"].mean()) if len(ok) else 0.0,
    }
    summary_path = out_dir / f"{kind}_{name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"✔ Saved {csv_path.name} and {summary_path.name}")


if __name__ == "__main__":
    main()
