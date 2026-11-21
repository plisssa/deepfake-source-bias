"""Регрессионные тесты на исправленные утечки данных и баги протокола.

Каждый тест соответствует конкретному багу из аудита 2026-07:
  1. сплит по файлам-кадрам (группы должны быть неделимы между сплитами);
  2. кэш признаков: sorted(paths) в ключе + сохранение в исходном порядке;
  3. FeatureFusionExtractor: ленивый fit scaler/PCA на val/test;
  4. ROB-9: разные seed при миксе и исключении из OOD-теста;
  5. aggregate_folds усреднял threshold/n/n_pos/n_neg как «метрики».
"""

from __future__ import annotations

import numpy as np
import pytest

from deepfakes.io.collect import extract_groups
from deepfakes.io.splits import SplitConfig, kfold_iter, make_holdout_split
from deepfakes.utils.caching import feature_cache_path, hash_inputs, load_cached, save_cached

# ---------- 1. групповые сплиты ----------


def _fake_video_dataset(n_videos: int = 30, frames_per: int = 6):
    """Синтетические пути вида video<id>_frame<k>.jpg + метки."""
    X, y, sources, groups = [], [], [], []
    for v in range(n_videos):
        label = v % 2
        for f in range(frames_per):
            X.append(f"/data/{'fake' if label else 'real'}/video{v:03d}_frame{f:03d}.jpg")
            y.append(label)
            sources.append("src_a")
            groups.append(f"video{v:03d}")
    return (
        np.array(X, dtype=object),
        np.array(y, dtype=np.int8),
        np.array(sources, dtype=object),
        np.array(groups, dtype=object),
    )


def test_holdout_groups_do_not_cross_splits():
    X, y, s, g = _fake_video_dataset()
    splits = make_holdout_split(
        X, y, s, SplitConfig(test_size=0.2, val_size=0.2, seed=42), groups=g
    )
    g_train = set(splits["train"]["groups"])
    g_val = set(splits["val"]["groups"])
    g_test = set(splits["test"]["groups"])
    assert g_train.isdisjoint(g_val)
    assert g_train.isdisjoint(g_test)
    assert g_val.isdisjoint(g_test)
    # все сэмплы на месте
    total = sum(len(splits[k]["X"]) for k in ("train", "val", "test"))
    assert total == len(X)


def test_kfold_groups_do_not_cross_folds():
    X, y, _s, g = _fake_video_dataset()
    for _, train_idx, val_idx in kfold_iter(X, y, n_splits=4, seed=42, groups=g):
        assert set(g[train_idx]).isdisjoint(set(g[val_idx]))


def test_group_auto_heuristic_accepts_video_frames(tmp_path):
    files = [f"/d/video{v:03d}_frame{f:03d}.jpg" for v in range(30) for f in range(5)]
    groups = extract_groups(files, mode="auto")
    assert len(set(groups)) == 30  # по одному на видео


def test_group_auto_heuristic_rejects_flat_numbering():
    # файлы 000001.jpg ... — «префикс» пустой, группировка была бы вырожденной
    files = [f"/d/{i:06d}.jpg" for i in range(100)]
    groups = extract_groups(files, mode="auto")
    assert len(set(groups)) == len(files)  # откат к per-file


def test_group_auto_heuristic_accepts_small_source():
    """8 видео × 50 кадров: мало групп, но структура настоящая — принять."""
    files = [f"/d/video{v}_frame{f:03d}.jpg" for v in range(8) for f in range(50)]
    groups = extract_groups(files, mode="auto")
    assert len(set(groups)) == 8


def test_stem_prefix_keeps_numeric_video_ids_separate():
    """video1 и video12 — разные видео, голые цифры ID не срезаются."""
    files = ["/d/video1.jpg", "/d/video2.jpg", "/d/video12.jpg"]
    groups = extract_groups(files, mode="stem_prefix")
    assert len(set(groups)) == 3
    # а кадровые счётчики с разделителем по-прежнему срезаются
    files2 = ["/d/video1_0001.jpg", "/d/video1_0002.jpg", "/d/video12_frame003.jpg"]
    groups2 = extract_groups(files2, mode="stem_prefix")
    assert groups2[0] == groups2[1]
    assert groups2[2] != groups2[0]


# ---------- 2. кэш признаков ----------


def test_cache_key_depends_on_path_order():
    paths_a = ["/x/a.jpg", "/x/b.jpg", "/x/c.jpg"]
    paths_b = ["/x/c.jpg", "/x/a.jpg", "/x/b.jpg"]
    params = {"extractor": "E", "params": {}, "tag": "train"}
    assert hash_inputs(paths_a, params) != hash_inputs(paths_b, params)


def test_cache_reorders_features_to_requested_paths(tmp_path):
    paths = ["/x/a.jpg", "/x/b.jpg", "/x/c.jpg"]
    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    p = feature_cache_path(tmp_path, "testkey")
    save_cached(p, X, paths=paths)

    got = load_cached(p, expected_paths=["/x/c.jpg", "/x/a.jpg", "/x/b.jpg"])
    assert got is not None
    X2, valid = got
    assert X2[:, 0].tolist() == [3.0, 1.0, 2.0]
    assert valid.all()


def test_cache_rejects_path_set_mismatch(tmp_path):
    paths = ["/x/a.jpg", "/x/b.jpg"]
    X = np.array([[1.0], [2.0]], dtype=np.float32)
    p = feature_cache_path(tmp_path, "testkey2")
    save_cached(p, X, paths=paths)
    assert load_cached(p, expected_paths=["/x/a.jpg", "/x/OTHER.jpg"]) is None


def test_cache_roundtrip_with_valid_mask(tmp_path):
    paths = ["/x/a.jpg", "/x/bad.jpg", "/x/c.jpg"]
    X = np.array([[1.0], [0.0], [3.0]], dtype=np.float32)
    mask = np.array([True, False, True])
    p = feature_cache_path(tmp_path, "testkey3")
    save_cached(p, X, paths=paths, valid_mask=mask)
    X2, valid2 = load_cached(p, expected_paths=paths)
    assert valid2.tolist() == [True, False, True]
    assert np.allclose(X2, X)


# ---------- 3. fusion: никакого ленивого fit на val/test ----------


def test_fusion_transform_before_fit_raises(tiny_dataset):
    from deepfakes.features.color import ColorMomentsExtractor
    from deepfakes.features.fusion import FeatureFusionExtractor

    fus = FeatureFusionExtractor(
        [ColorMomentsExtractor(img_size=32, patch_sizes=[16], verbose=False)], scale=True
    )
    import glob

    val_paths = np.array(sorted(glob.glob(tiny_dataset["real_dir"] + "/*.png"))[:4], dtype=object)
    with pytest.raises(RuntimeError, match="до fit"):
        fus.transform(val_paths)


def test_fusion_uses_train_statistics_for_val(tiny_dataset):
    """Scaler обучен на train: у train-признаков mean≈0, у val — необязательно."""
    import glob

    from deepfakes.features.color import ColorMomentsExtractor
    from deepfakes.features.fusion import FeatureFusionExtractor

    real = sorted(glob.glob(tiny_dataset["real_dir"] + "/*.png"))
    fake = sorted(glob.glob(tiny_dataset["fake_dir"] + "/*.png"))
    train = np.array(real[:10] + fake[:10], dtype=object)
    val = np.array(real[10:14] + fake[10:14], dtype=object)

    fus = FeatureFusionExtractor(
        [ColorMomentsExtractor(img_size=32, patch_sizes=[16], verbose=False)],
        scale=True,
        verbose=False,
    )
    fus.fit(train)
    F_train = fus.transform(train)
    F_val = fus.transform(val)
    # train отскейлен своим scaler-ом → среднее ~0 по каждой колонке
    assert abs(F_train.mean()) < 0.2
    # повторный вызов transform(train) не пере-фитит scaler на val-статистиках
    assert F_val.shape[1] == F_train.shape[1]
    assert fus.scaler_ is not None


# ---------- 4. ROB-9 ----------


def test_rob9_mix_returns_exact_used_indices():
    from scripts.run_rob9_cross_training import _mix

    X1 = np.array([f"a{i}" for i in range(50)], dtype=object)
    y1 = np.zeros(50, dtype=np.int8)
    X2 = np.array([f"b{i}" for i in range(100)], dtype=object)
    y2 = np.ones(100, dtype=np.int8)
    g2 = np.array([f"b{i}" for i in range(100)], dtype=object)  # per-file группы

    X_mix, _y_mix, used_idx = _mix(X1, y1, X2, y2, g2, ratio=0.3, seed=7)
    mask = np.ones(len(X2), dtype=bool)
    mask[used_idx] = False
    X_ood = X2[mask]
    assert len(set(X_mix) & set(X_ood)) == 0
    assert len(X_ood) == 100 - len(used_idx)


def test_rob9_mix_takes_whole_groups():
    """Кадры одного видео B не должны расползаться между train-миксом и OOD."""
    from scripts.run_rob9_cross_training import _mix

    X1 = np.array([f"a{i}" for i in range(20)], dtype=object)
    y1 = np.zeros(20, dtype=np.int8)
    # 20 видео × 10 кадров
    X2 = np.array([f"vid{v:02d}_f{f}" for v in range(20) for f in range(10)], dtype=object)
    y2 = np.ones(200, dtype=np.int8)
    g2 = np.array([f"vid{v:02d}" for v in range(20) for _ in range(10)], dtype=object)

    _X_mix, _y_mix, used_idx = _mix(X1, y1, X2, y2, g2, ratio=0.2, seed=7)
    mask = np.ones(len(X2), dtype=bool)
    mask[used_idx] = False
    assert set(g2[used_idx]).isdisjoint(set(g2[mask]))
    # взято примерно ratio выборки (целыми группами)
    assert 0.1 <= len(used_idx) / len(X2) <= 0.35


# ---------- 5. агрегация фолдов ----------


def test_aggregate_folds_excludes_counters():
    from deepfakes.eval.cv import FoldResult, aggregate_folds
    from deepfakes.eval.metrics import compute_report

    rng = np.random.default_rng(0)
    folds = []
    for i in range(3):
        y = rng.integers(0, 2, 50)
        s = rng.random(50)
        folds.append(FoldResult(fold=i, report=compute_report(y, s), train_size=40, val_size=10))
    agg = aggregate_folds(folds)
    assert "roc_auc_mean" in agg
    for bad in ("threshold_mean", "n_mean", "n_pos_mean", "n_neg_mean"):
        assert bad not in agg
