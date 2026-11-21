"""Контрольные наборы для статьи: воспроизведение схемы A и контроля «только обработка».

1. ``a2`` — схема A (реальные CelebA против синтезированной половины 140k) с
   разбиением, в котором ЛИЧНОСТИ НЕ ПЕРЕСЕКАЮТСЯ: реальные берутся из
   официальных разделов CelebA (train/val/test = разделы 0/1/2, личности в них
   различны). Синтезированные — ровно те же 20 000 файлов и тот же сплит, что
   в июльском прогоне A и в согласованной схеме gan_only v1 (проверено
   29.09.2026: 20 000 из 20 000 совпадают, сплит тот же). Поэтому новая A и
   согласованная D различаются ТОЛЬКО реальным классом.
   В июльском прогоне групповой сплит был неактивен, и одна знаменитость
   попадала и в train, и в test (RESULTS_2026-07-30 §3).

2. ``resample`` — контроль «только обработка» для схемы A (воспроизведение
   июльского эксперимента «1620 фотографий», выполненного вне конвейера и
   задокументированного не полностью). Реальные фото одного источника
   (140k valid/real — FFHQ, нигде больше не используются) делятся на две
   непересекающиеся группы и готовятся двумя способами:
       as_real_A : 256×256 → 178×218 (геометрия CelebA) → JPEG q95;
                   конвейер признаков затем вернёт их к квадрату — как CelebA
       as_fake_A : 256×256 → JPEG q95 без изменения размера — как 140k
   Обе группы пережаты ОДИН раз одним энкодером: различается только геометрия
   подготовки. Синтезированных изображений нет вовсе, поэтому любое качество
   выше 0,5 — вклад обработки.

    python -m scripts.make_control_sets a2
    python -m scripts.make_control_sets resample --out data/raw/ctrl_resample
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data/splits"
CELEBA = REPO / "data/raw/celeba"
R140K = REPO / "data/raw/real_fake_140k/real_vs_fake/real-vs-fake"
N_REAL = {"train": 14000, "val": 3000, "test": 3000}
PART = {"train": 0, "val": 1, "test": 2}
CELEBA_SIZE = (178, 218)


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def celeba_by_partition(partition_csv: Path, img_dir: Path, seed: int = 42) -> dict[str, list[str]]:
    """Реальные CelebA по официальным разделам; внутри раздела — детерминированная выборка."""
    part = pd.read_csv(partition_csv)
    rng = np.random.default_rng(seed)
    out = {}
    for split, k in PART.items():
        ids = part.loc[part.partition == k, "image_id"].to_numpy()
        pick = rng.choice(len(ids), size=N_REAL[split], replace=False)
        out[split] = [str(img_dir / ids[i]) for i in sorted(pick)]
    return out


def write_a2(fake_split_dir: Path, out_dir: Path, reals: dict[str, list[str]]) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        fk = pd.read_csv(fake_split_dir / f"{split}.csv")
        fk = fk[fk.label == 1]
        rl = pd.DataFrame(
            {
                "path": reals[split],
                "label": 0,
                "source": "celeba_real",
                "group": [f"celeba_part{PART[split]}::{Path(p).name}" for p in reals[split]],
            }
        )
        df = pd.concat([rl, fk[["path", "label", "source", "group"]]], ignore_index=True)
        df.to_csv(out_dir / f"{split}.csv", index=False)
        counts[split] = len(df)
    return counts


def prepare_as_real_a(src: Path, dst: Path) -> None:
    """Геометрия CelebA: 178×218 (с изменением пропорций), JPEG q95."""
    im = Image.open(src).convert("RGB").resize(CELEBA_SIZE, Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="JPEG", quality=95)


def prepare_as_fake_a(src: Path, dst: Path) -> None:
    """Геометрия 140k: 256×256 без изменения размера, то же перекодирование q95."""
    im = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="JPEG", quality=95)


def split_of(name: str) -> str:
    b = _h(f"resample-split:{name}") % 100
    return "train" if b < 70 else "val" if b < 85 else "test"


def build_resample(src_dir: Path, out: Path, n_per_group: int, splits_out: Path) -> dict:
    files = sorted(src_dir.glob("*.jpg"), key=lambda p: _h(f"resample-pick:{p.name}"))
    files = files[: 2 * n_per_group]
    rows = []
    for i, f in enumerate(files):
        # чередование по порядку хэша: две непересекающиеся группы одного источника
        grp = "as_real_A" if i % 2 == 0 else "as_fake_A"
        dst = out / grp / f.name
        if not dst.exists():
            (prepare_as_real_a if grp == "as_real_A" else prepare_as_fake_a)(f, dst)
        rows.append(
            {
                "path": str(dst),
                "label": int(grp == "as_fake_A"),
                "source": grp,
                "group": f"ffhq_valid::{f.stem}",
                "split": split_of(f.name),
            }
        )
    df = pd.DataFrame(rows)
    splits_out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        df[df.split == s][["path", "label", "source", "group"]].to_csv(splits_out / f"{s}.csv", index=False)
    return df.groupby(["source", "split"]).size().unstack(fill_value=0).to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("a2")
    r = sub.add_parser("resample")
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=5000, help="изображений в каждой группе")
    args = ap.parse_args()
    if args.cmd == "a2":
        reals = celeba_by_partition(CELEBA / "list_eval_partition.csv", CELEBA / "img_align_celeba/img_align_celeba")
        print(write_a2(SPLITS / "gan_only/v1", SPLITS / "celeba_vs_140k/v2", reals))
    else:
        print(build_resample(R140K / "valid/real", Path(args.out), args.n, SPLITS / "ctrl_resample/v1"))


if __name__ == "__main__":
    main()
