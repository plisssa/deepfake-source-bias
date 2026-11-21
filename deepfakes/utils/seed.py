"""Единая фиксация сидов для воспроизводимости."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> int:
    """Зафиксировать random_state для всех релевантных библиотек.

    Returns:
        Использованный сид.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import cv2

        cv2.setRNGSeed(seed)
    except ImportError:
        pass

    return seed
