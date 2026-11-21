"""Дубликаты изображений по разные стороны сплита — утечка, которую не ловят группы.

Групповые сплиты (по человеку, по видео) защищают от общих личностей, но не от
того, что ОДНА И ТА ЖЕ фотография лежит в датасете дважды или в виде
пересохранённой копии. Если такая пара попадает в train и test, модель видит
тестовый объект при обучении.

Два уровня:
* **MD5 байтов** — точные копии файлов;
* **dHash 64 бит** (разностный хэш по уменьшенному до 9×8 серому кадру) —
  визуальные копии: переживает пересжатие JPEG и небольшой ресайз. Совпадение
  dHash (расстояние 0) — сильный признак той же картинки; для контроля ложных
  совпадений печатается, одинакова ли у пары метка класса.

Только чтение файлов (PIL + numpy + stdlib), MLflow не трогает.

    python -m scripts.check_duplicates data=gan_only
    python -m scripts.check_duplicates data=dm_only
Код выхода: 0 — межсплитовых дубликатов нет, 2 — есть.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image

SPLITS = ("train", "val", "test")


def fingerprint(path: str) -> tuple[str, int | None]:
    """(md5 байтов, dHash) — dHash=None, если файл не читается."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return "", None
    md5 = hashlib.md5(data).hexdigest()
    try:
        from io import BytesIO

        with Image.open(BytesIO(data)) as img:
            g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        dh = int("".join("1" if b else "0" for b in bits), 2)
    except Exception:
        dh = None
    return md5, dh


def cross_split_pairs(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Группы с одинаковым ключом, встречающиеся более чем в одном сплите."""
    out = []
    for k, g in df.dropna(subset=[key]).groupby(key):
        if g["split"].nunique() > 1:
            out.append(
                {
                    "key": k,
                    "n": len(g),
                    "splits": ",".join(sorted(g["split"].unique())),
                    "labels": ",".join(str(x) for x in sorted(g["label"].unique())),
                    "example": g["path"].iloc[0],
                }
            )
    return pd.DataFrame(out)


def _small_gray(path: str) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)
    except Exception:
        return None


def confirm_visual(
    df: pd.DataFrame, candidates: pd.DataFrame, max_mad: float = 4.0
) -> pd.DataFrame:
    """Подтвердить совпадения dHash попиксельно.

    dHash строится по сетке 9×8 и СТАЛКИВАЕТСЯ у разных изображений со сходной
    грубой структурой — а лица в наших датасетах выровнены одинаково. Поэтому
    совпадение хэша — только кандидат. Дубликатом считается пара из РАЗНЫХ
    сплитов со средней абсолютной разницей < max_mad (шкала 0–255) на 64×64."""
    rows = []
    for key in candidates["key"]:
        g = df[df["dhash"] == key]
        imgs = {r.path: _small_gray(r.path) for r in g.itertuples()}
        best = None
        for a in g.itertuples():
            for b in g.itertuples():
                if a.split >= b.split or imgs[a.path] is None or imgs[b.path] is None:
                    continue
                mad = float(np.abs(imgs[a.path] - imgs[b.path]).mean())
                if best is None or mad < best[0]:
                    best = (mad, a.path, b.path, a.label, b.label, a.split, b.split)
        if best is not None:
            rows.append(
                {
                    "mad": round(best[0], 2),
                    "confirmed": best[0] < max_mad,
                    "path_a": best[1],
                    "path_b": best[2],
                    "labels": f"{best[3]},{best[4]}",
                    "splits": f"{best[5]},{best[6]}",
                }
            )
    return pd.DataFrame(rows)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Тот же вход, что у остальных скриптов. Первая версия собирала конфиг через
    # compose() без контекста Hydra, и подстановки ${hydra:...} в paths падали с
    # `HydraConfig was not set` (29.09.2026).
    split_dir = Path(cfg.paths.splits_dir) / cfg.data.name / str(cfg.data.splits_version)
    df = pd.concat(
        [pd.read_csv(split_dir / f"{s}.csv").assign(split=s) for s in SPLITS], ignore_index=True
    )
    with ProcessPoolExecutor(max_workers=8) as ex:
        fps = list(ex.map(fingerprint, df["path"], chunksize=256))
    df["md5"] = [f[0] or None for f in fps]
    df["dhash"] = [f[1] for f in fps]

    print(f"=== {cfg.data.name} {cfg.data.splits_version}: {len(df)} изображений")
    print(f"  не прочитано: {int(df['dhash'].isna().sum())}")
    within = defaultdict(int)
    for s in SPLITS:
        d = df[df.split == s]
        within[s] = int(d["md5"].duplicated().sum())
    print(f"  точные копии ВНУТРИ сплитов (не утечка, но шум): {dict(within)}")

    exact = cross_split_pairs(df, "md5")
    visual = cross_split_pairs(df, "dhash")
    print(f"  ТОЧНЫЕ копии между сплитами (MD5): {len(exact)}")
    print(f"  ВИЗУАЛЬНЫЕ совпадения между сплитами (dHash=0): {len(visual)}")
    confirmed = pd.DataFrame()
    if len(visual):
        checked = confirm_visual(df, visual)
        confirmed = checked[checked["confirmed"]] if len(checked) else checked
        print(
            f"    из них ПОДТВЕРЖДЕНО попиксельно (MAD < 4 на 64×64): {len(confirmed)} "
            f"— остальное ложные совпадения хэша"
        )
        if len(confirmed):
            same = int(
                (
                    confirmed["labels"].str.split(",").str[0]
                    == confirmed["labels"].str.split(",").str[1]
                ).sum()
            )
            print(
                f"    подтверждённые с одинаковой меткой: {same}, с разной: {len(confirmed) - same}"
            )
            print(confirmed.head(8).to_string(index=False))
        if len(checked):
            print(
                f"    MAD у кандидатов: медиана {checked['mad'].median():.1f}, минимум {checked['mad'].min():.1f}"
            )
    out = Path(cfg.paths.reports_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.data.name}_{cfg.data.splits_version}"
    exact.to_csv(out / f"duplicates_exact_{tag}.csv", index=False)
    visual.to_csv(out / f"duplicates_dhash_{tag}.csv", index=False)
    confirmed.to_csv(out / f"duplicates_confirmed_{tag}.csv", index=False)
    sys.exit(2 if len(exact) or len(confirmed) else 0)


if __name__ == "__main__":
    main()
