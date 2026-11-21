"""Сплиты по ЛИЧНОСТИ, когда один человек есть в нескольких источниках.

Регрессия 28.09.2026: DeepFakeFace генерирует фейки ИЗ wiki-фотографий, поэтому
у человека есть реальное фото и до трёх фейков разных семейств. Сплит v2
группировал по файлу (к тому же к группе дописывалось имя источника), и 36.1%
фейков в test принадлежали людям, чьи фейки уже были в train с той же меткой —
неявная утечка личности (Dong et al., CVPR 2023).
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from PIL import Image

from deepfakes.io.collect import DatasetBundle, DatasetSource
from deepfakes.io.splits import SplitConfig, make_holdout_split

RX = r"/\d{6}_(\d+)_\d{4}-\d{2}-\d{2}_\d{4}\.jpg$"
PID = re.compile(r"/\d{6}_(\d+)_")


def _mk_dataset(root, n_people=60):
    """wiki (real) + два семейства фейков: у каждого человека фото в КАЖДОМ источнике.

    Имена — в формате normalize_domain.py: <индекс>_<id>_<дата рождения>_<год>.jpg,
    индексы в разных источниках НЕ совпадают (как на кластере)."""
    rng = np.random.default_rng(0)
    for src, offset in (("wiki", 0), ("text2img", 500), ("inpainting", 900)):
        d = root / src
        d.mkdir(parents=True)
        for k in range(n_people):
            pid = 1000 + k
            arr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{offset + k:06d}_{pid}_1980-01-01_2010.jpg")
    return root


def _src(root, name, label, **kw):
    return DatasetSource(name=name, path=str(root / name), label=label, use_subfolders=False, **kw)


GROUP = dict(group_mode="regex", group_regex=RX, group_namespace="wiki_person")


def test_shared_namespace_puts_person_in_one_group(tmp_path):
    root = _mk_dataset(tmp_path)
    b = DatasetBundle(
        real_sources=[_src(root, "wiki", 0, **GROUP)],
        fake_sources=[_src(root, "text2img", 1, **GROUP), _src(root, "inpainting", 1, **GROUP)],
    )
    X, _y, _sources, groups = b.collect()
    by_pid: dict[str, set] = {}
    for x, g in zip(X, groups):
        by_pid.setdefault(PID.search(str(x)).group(1), set()).add(g)
    assert all(len(v) == 1 for v in by_pid.values()), "у человека должна быть ОДНА группа"
    assert len(by_pid) == 60


def test_default_namespace_keeps_sources_apart(tmp_path):
    """Без явного пространства имён прежнее поведение: источники не склеиваются."""
    root = _mk_dataset(tmp_path)
    kw = dict(group_mode="regex", group_regex=RX)
    _X, _y, _s, groups = DatasetBundle(
        real_sources=[_src(root, "wiki", 0, **kw)],
        fake_sources=[_src(root, "text2img", 1, **kw)],
    ).collect()
    assert len(set(groups)) == 120  # 60 людей × 2 источника


def test_split_has_no_person_on_both_sides(tmp_path):
    """Главная проверка: ни один человек не попадает и в train, и в test."""
    root = _mk_dataset(tmp_path, n_people=200)
    X, y, sources, groups = DatasetBundle(
        real_sources=[_src(root, "wiki", 0, **GROUP)],
        fake_sources=[_src(root, "text2img", 1, **GROUP), _src(root, "inpainting", 1, **GROUP)],
    ).collect()
    sp = make_holdout_split(
        X, y, sources, SplitConfig(test_size=0.2, val_size=0.2, seed=1), groups=groups
    )
    people = {k: {PID.search(str(p)).group(1) for p in v["X"]} for k, v in sp.items()}
    assert not people["train"] & people["test"]
    assert not people["train"] & people["val"]
    assert not people["val"] & people["test"]


def test_per_file_grouping_would_leak(tmp_path):
    """Контрольный: при группировке по файлу утечка действительно возникает —
    иначе тест выше ничего бы не доказывал."""
    root = _mk_dataset(tmp_path, n_people=200)
    X, y, sources, groups = DatasetBundle(
        real_sources=[_src(root, "wiki", 0, group_mode="none")],
        fake_sources=[
            _src(root, "text2img", 1, group_mode="none"),
            _src(root, "inpainting", 1, group_mode="none"),
        ],
    ).collect()
    sp = make_holdout_split(
        X, y, sources, SplitConfig(test_size=0.2, val_size=0.2, seed=1), groups=groups
    )
    tr = {PID.search(str(p)).group(1) for p in sp["train"]["X"]}
    te = {PID.search(str(p)).group(1) for p in sp["test"]["X"]}
    assert tr & te, "при группировке по файлу личности должны пересекаться"


def test_hash_split_by_group_keeps_person_on_one_side(tmp_path):
    """even/odd по личности: все файлы человека — в одной половине, в ЛЮБОМ источнике."""
    root = _mk_dataset(tmp_path, n_people=100)
    halves = {}
    for half in ("even", "odd"):
        people = set()
        for src, lab in (("wiki", 0), ("text2img", 1), ("inpainting", 1)):
            s = _src(root, src, lab, hash_split=half, hash_split_by="group", **GROUP)
            files, _, _ = s.collect()
            people |= {PID.search(f).group(1) for f in files}
        halves[half] = people
    assert halves["even"] and halves["odd"]
    assert not halves["even"] & halves["odd"]
    assert len(halves["even"] | halves["odd"]) == 100


def test_hash_split_by_file_mixes_people_across_sources(tmp_path):
    """Контрольный: прежний режим (по имени файла) людей между половинами мешает."""
    root = _mk_dataset(tmp_path, n_people=100)
    even_real = {
        PID.search(f).group(1)
        for f in _src(root, "wiki", 0, hash_split="even", **GROUP).collect()[0]
    }
    odd_fake = {
        PID.search(f).group(1)
        for f in _src(root, "text2img", 1, hash_split="odd", **GROUP).collect()[0]
    }
    assert even_real & odd_fake, "по имени файла половины людей пересекаются"


def test_bad_hash_split_by_is_rejected(tmp_path):
    root = _mk_dataset(tmp_path, n_people=4)
    with pytest.raises(ValueError):
        _src(root, "wiki", 0, hash_split="even", hash_split_by="person").collect()
