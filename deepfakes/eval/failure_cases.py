"""Top-N false positives / false negatives → отчёт с превью изображений."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


def collect_failure_cases(
    paths: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_dir: str | Path,
    n_top: int = 20,
    threshold: float = 0.5,
    copy_images: bool = False,
) -> dict[str, Path]:
    """Сохранить top-N FP и FN с превью.

    Args:
        copy_images: если True — копирует сами изображения в out_dir/{fp,fn}/.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_score >= threshold).astype(int)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    # FP: real, который модель посчитала fake. Берём с самым высоким score.
    fp_idx = np.argsort(y_score)[::-1]
    fp_idx = fp_idx[fp_mask[fp_idx]][:n_top]
    # FN: fake, который модель посчитала real. Берём с самым низким score.
    fn_idx = np.argsort(y_score)
    fn_idx = fn_idx[fn_mask[fn_idx]][:n_top]

    paths_str = np.array([str(p) for p in paths])

    fp_df = pd.DataFrame(
        {"path": paths_str[fp_idx], "y_true": y_true[fp_idx], "y_score": y_score[fp_idx]}
    )
    fn_df = pd.DataFrame(
        {"path": paths_str[fn_idx], "y_true": y_true[fn_idx], "y_score": y_score[fn_idx]}
    )
    fp_csv = out_dir / "top_false_positives.csv"
    fn_csv = out_dir / "top_false_negatives.csv"
    fp_df.to_csv(fp_csv, index=False)
    fn_df.to_csv(fn_csv, index=False)

    if copy_images:
        for sub, df in [("fp", fp_df), ("fn", fn_df)]:
            sd = out_dir / sub
            sd.mkdir(exist_ok=True)
            for _, row in df.iterrows():
                src = Path(row["path"])
                if src.exists():
                    shutil.copy(src, sd / src.name)

    log.info(f"Saved {len(fp_df)} FP and {len(fn_df)} FN to {out_dir}")
    return {"fp_csv": fp_csv, "fn_csv": fn_csv}
