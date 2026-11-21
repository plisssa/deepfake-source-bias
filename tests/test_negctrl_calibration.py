"""Калибровка негативного контроля: он обязан ВИДЕТЬ тривиальный канал известной силы.

Регрессия 28.09.2026: прежняя реализация (RandomForest с листьями до одного
объекта) тянула к 0.5. Канал с истинным AUC 0.646 она показывала как 0.553 —
то есть контроль «проходил» порог 0.6, хотя тривиальный признак реально
разделял классы. На GAN-домене вес файла давал 0.5517, контроль — 0.497.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from deepfakes.eval.negative_control import _fit_auc


def _scalar_channel(shift: float, seed: int):
    rng = np.random.default_rng(seed)

    def draw(n):
        y = rng.integers(0, 2, n)
        x = rng.normal(0, 1, n) - shift * y  # фейк «легче», как вес файла
        return (x * 4000 + 30000).astype(np.float32).reshape(-1, 1), y

    return (*draw(20000), *draw(6000))


@pytest.mark.parametrize("shift", [0.18, 0.5, 1.8])
def test_control_recovers_known_scalar_signal(shift):
    Xtr, ytr, Xva, yva = _scalar_channel(shift, seed=int(shift * 100))
    true_auc = roc_auc_score(yva, -Xva[:, 0])
    got = _fit_auc(Xtr, ytr, Xva, yva)
    assert got >= true_auc - 0.02, f"контроль занижает: истинный {true_auc:.3f}, дал {got:.3f}"


def test_moderate_channel_no_longer_passes_threshold():
    """Главный сценарий ошибки: канал ~0.65 не должен выглядеть как «< 0.6»."""
    Xtr, ytr, Xva, yva = _scalar_channel(0.5, seed=7)
    assert _fit_auc(Xtr, ytr, Xva, yva) >= 0.6


def test_no_signal_stays_near_half():
    """И обратно: на чистом шуме контроль не должен выдумывать канал."""
    Xtr, ytr, Xva, yva = _scalar_channel(0.0, seed=11)
    assert _fit_auc(Xtr, ytr, Xva, yva) < 0.53


def test_nonmonotone_group_is_detected():
    """Двумерная группа с нелинейной связью (XOR-подобной) должна ловиться бустингом."""
    rng = np.random.default_rng(3)

    def draw(n):
        a, b = rng.normal(size=n), rng.normal(size=n)
        y = ((a > 0) ^ (b > 0)).astype(int)
        return np.c_[a, b].astype(np.float32), y

    Xtr, ytr = draw(20000)
    Xva, yva = draw(6000)
    assert _fit_auc(Xtr, ytr, Xva, yva) > 0.9


def test_direction_is_chosen_on_train_not_val():
    """Направление рангового AUC — по train: val-метки не должны подсказывать знак."""
    Xtr, ytr, Xva, yva = _scalar_channel(1.8, seed=5)
    flipped = _fit_auc(Xtr, ytr, -Xva, yva)  # val перевёрнут относительно train
    # если бы знак выбирался по val, flipped остался бы высоким
    assert flipped < 0.6
