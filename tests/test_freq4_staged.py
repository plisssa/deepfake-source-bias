"""FREQ-4 staged/continual обучение работает и завершается одной финальной моделью."""

from __future__ import annotations

import pytest

from deepfakes.models.classical import ModelSpec
from deepfakes.pipelines.freq4 import Freq4Pipeline, _lr_schedule


def test_lr_schedule_warmup_first_stage_lower():
    base = 0.1
    lr_warmup = _lr_schedule(0, 3, base, "warmup_cosine")
    lr_mid = _lr_schedule(1, 3, base, "warmup_cosine")
    assert lr_warmup < base, "First stage should be warmup (lr < base)"
    assert lr_mid > 0


def test_lr_schedule_step_decay_monotonic():
    base = 0.1
    a = _lr_schedule(0, 4, base, "step_decay")
    b = _lr_schedule(1, 4, base, "step_decay")
    c = _lr_schedule(2, 4, base, "step_decay")
    assert a > b > c > 0


def test_lr_schedule_none_returns_base():
    assert _lr_schedule(2, 5, 0.1, "none") == 0.1


@pytest.mark.parametrize("staged_mode", ["none", "staged", "continual"])
def test_freq4_three_modes_run(tiny_splits, staged_mode):
    cfg = {
        "img_size": 64,
        "n_bins": 6,
        "include_slope": True,
        "include_flatness": True,
        "include_basic_stats": True,
        "use_gan_augment": False,
        "staged_mode": staged_mode,
        "n_stages": 2,
        "lr_schedule": "warmup_cosine",
    }
    pipe = Freq4Pipeline.from_cfg(cfg, ModelSpec(name="xgboost", params={}))
    pipe.fit(
        tiny_splits["train"]["X"],
        tiny_splits["train"]["y"],
        X_val_paths=tiny_splits["val"]["X"],
        y_val=tiny_splits["val"]["y"],
    )
    probs = pipe.predict_proba(tiny_splits["test"]["X"])
    assert probs.shape == (len(tiny_splits["test"]["X"]),)
    assert (probs >= 0).all() and (probs <= 1).all()
