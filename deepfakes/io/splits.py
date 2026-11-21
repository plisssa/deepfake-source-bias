"""Стратифицированный сплит и сохранение/загрузка фиксированных сплитов.

Защита от утечек:
  * если у сэмплов есть группы (video_id / identity / префикс кадра), сплит
    выполняется ПО ГРУППАМ — одна группа целиком попадает ровно в один из
    train/val/test (StratifiedGroupKFold);
  * стратификация по y×source, чтобы пропорции источников (с разными
    разрешениями/JPEG-качеством) не разъезжались между сплитами.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15  # доля от ИСХОДНОГО датасета, не от trainval
    seed: int = 42
    stratify: bool = True
    stratify_by_source: bool = True  # стратифицировать по y×source, а не только по y


def _strat_labels(y: np.ndarray, sources: np.ndarray, cfg: SplitConfig) -> np.ndarray | None:
    if not cfg.stratify:
        return None
    if cfg.stratify_by_source:
        return np.array([f"{int(lbl)}|{src}" for lbl, src in zip(y, sources)], dtype=object)
    return y


def _has_real_groups(groups: np.ndarray | None) -> bool:
    return groups is not None and len(np.unique(groups)) < len(groups)


def _group_split_once(
    X: np.ndarray,
    y: np.ndarray,
    strat: np.ndarray | None,
    groups: np.ndarray,
    holdout_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Отделить ~holdout_size групповой стратифицированный холдаут.

    StratifiedGroupKFold, холдаут = объединение k из n фолдов, где k/n ближе
    всего к holdout_size (round(1/size) одним фолдом давал грубые пропорции:
    0.4 -> 50%). Стратификация при групповом сплите приближённая (группы
    неделимы).
    """
    from fractions import Fraction

    n_groups = len(np.unique(groups))
    frac = Fraction(holdout_size).limit_denominator(20)
    n_splits = min(max(2, frac.denominator), n_groups)
    k = max(1, round(holdout_size * n_splits))
    y_for_strat = strat if strat is not None else y
    # категориальные метки → коды для sklearn
    _, y_codes = np.unique(y_for_strat, return_inverse=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [test for _, test in sgkf.split(X, y_codes, groups=groups)]
    holdout_idx = np.concatenate(folds[:k])
    mask = np.ones(len(X), dtype=bool)
    mask[holdout_idx] = False
    rest_idx = np.nonzero(mask)[0]
    return rest_idx, holdout_idx


def make_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    cfg: SplitConfig,
    groups: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Стратифицированный train/val/test = (1 - val - test) / val / test.

    Если переданы содержательные groups (есть повторяющиеся) — групповой сплит:
    ни одна группа не делится между train/val/test.

    Returns:
        dict с ключами 'train', 'val', 'test',
        каждый = {'X', 'y', 'sources', 'groups'}.
    """
    if groups is None:
        groups = np.asarray(X, dtype=object)  # группа = сам файл (без ограничения)
    strat = _strat_labels(y, sources, cfg)

    if not _has_real_groups(groups):
        # Молчаливое вырождение опасно: для датасетов с несколькими снимками
        # одной личности (CelebA: ~10k личностей на 202k изображений) это
        # означает implicit identity leakage между train и test.
        log.warning(
            "Групповой сплит НЕ активен: у каждого сэмпла своя группа. "
            "Если в источнике несколько изображений одной личности/видео, "
            "они попадут и в train, и в test (implicit identity leakage). "
            "Задайте group_mode/group_regex в конфиге источника."
        )

    if _has_real_groups(groups):
        log.info(f"  group-aware split: {len(np.unique(groups))} groups / {len(X)} samples")
        rest_idx, test_idx = _group_split_once(X, y, strat, groups, cfg.test_size, cfg.seed)
        val_fraction_of_rest = cfg.val_size / (1.0 - cfg.test_size)
        strat_rest = strat[rest_idx] if strat is not None else None
        tr_rel, val_rel = _group_split_once(
            X[rest_idx],
            y[rest_idx],
            strat_rest,
            groups[rest_idx],
            val_fraction_of_rest,
            cfg.seed + 1,
        )
        train_idx, val_idx = rest_idx[tr_rel], rest_idx[val_rel]
    else:
        idx_all = np.arange(len(X))
        rest_idx, test_idx = train_test_split(
            idx_all,
            test_size=cfg.test_size,
            stratify=strat,
            random_state=cfg.seed,
        )
        val_fraction_of_rest = cfg.val_size / (1.0 - cfg.test_size)
        strat_rest = strat[rest_idx] if strat is not None else None
        train_idx, val_idx = train_test_split(
            rest_idx,
            test_size=val_fraction_of_rest,
            stratify=strat_rest,
            random_state=cfg.seed,
        )

    def _pack(idx: np.ndarray) -> dict[str, np.ndarray]:
        return {"X": X[idx], "y": y[idx], "sources": sources[idx], "groups": groups[idx]}

    splits = {"train": _pack(train_idx), "val": _pack(val_idx), "test": _pack(test_idx)}

    # Проверка «страховочным ремнём»: группы не пересекаются между сплитами.
    if _has_real_groups(groups):
        _check_group_overlap(splits)

    for name, sp in splits.items():
        pos = int((sp["y"] == 1).sum())
        neg = int((sp["y"] == 0).sum())
        log.info(f"  split[{name}] n={len(sp['y'])} (real={neg}, fake={pos})")
    return splits


def _check_group_overlap(splits: dict[str, dict[str, np.ndarray]]) -> None:
    """RuntimeError, если хоть одна группа встречается в двух сплитах."""
    g_tr, g_va, g_te = (set(splits[k]["groups"]) for k in ("train", "val", "test"))
    overlap = (g_tr & g_va) | (g_tr & g_te) | (g_va & g_te)
    if overlap:
        raise RuntimeError(f"Group leakage across splits: {sorted(overlap)[:5]} ...")


def save_split_csv(splits: dict[str, dict[str, np.ndarray]], out_dir: str | Path) -> None:
    """Сохранить сплит как train.csv / val.csv / test.csv (с группами)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sp in splits.items():
        df = pd.DataFrame(
            {
                "path": sp["X"],
                "label": sp["y"],
                "source": sp["sources"],
                "group": sp.get("groups", sp["X"]),
            }
        )
        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        log.info(f"  saved {out_path} (n={len(df)})")


def load_split_csv(in_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Загрузить сплит из train.csv / val.csv / test.csv.

    Валидация против устаревших сплитов: CSV, созданный ДО группового фикса
    (без колонки group), молча возвращал бы «дырявый» сплит — теперь об этом
    пишется громкий warning; при наличии групп проверяется их непересечение."""
    in_dir = Path(in_dir)
    splits: dict[str, dict[str, np.ndarray]] = {}
    missing_group_col = False
    for name in ("train", "val", "test"):
        p = in_dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Split CSV not found: {p}")
        df = pd.read_csv(p)
        if "group" not in df.columns:
            missing_group_col = True
        splits[name] = {
            "X": df["path"].to_numpy(),
            "y": df["label"].to_numpy(dtype=np.int8),
            "sources": df["source"].to_numpy(),
            "groups": (df["group"].to_numpy() if "group" in df.columns else df["path"].to_numpy()),
        }
    if missing_group_col:
        log.warning(
            f"Split CSVs at {in_dir} have no 'group' column (created before the "
            f"group-aware fix): frame-level leakage between splits is possible. "
            f"Delete them or bump data.splits_version to regenerate."
        )
    else:
        all_groups = np.concatenate([splits[k]["groups"] for k in ("train", "val", "test")])
        if _has_real_groups(all_groups):
            _check_group_overlap(splits)
    return splits


def kfold_iter(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    groups: np.ndarray | None = None,
):
    """Стратифицированный K-Fold итератор (групповой, если переданы groups).

    Возвращает (fold_idx, train_idx, val_idx)."""
    if _has_real_groups(groups):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = skf.split(X, y, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = skf.split(X, y)
    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        yield fold_idx, train_idx, val_idx
