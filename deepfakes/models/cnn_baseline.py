"""Neural baseline architectures supported by the training package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepfakes.utils.logging import get_logger

log = get_logger(__name__)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    # Полноценные заглушки: модуль обязан ИМПОРТИРОВАТЬСЯ без torch (registry
    # оборачивает импорт dl-пайплайнов в try/except ImportError, но NameError
    # от неопределённых Dataset/DataLoader он не ловил — dl тихо ломал импорт).
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


@dataclass
class DLConfig:
    backbone: str = "efficientnet_b0"  # 'efficientnet_b0' | 'vit_tiny_patch16_224' | 'resnet18'
    pretrained: bool = True
    num_classes: int = 2
    img_size: int = 224
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda' | 'mps'

    def resolve_device(self) -> str:
        if not TORCH_AVAILABLE:
            return "cpu"
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class _ImagePathsDataset(Dataset):
    """Минимальный Dataset поверх списка путей + меток. Использует torchvision transforms."""

    def __init__(self, paths, labels, img_size: int, train: bool = False):
        from torchvision import transforms

        self.paths = list(paths)
        self.labels = list(labels) if labels is not None else None
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((img_size + 32, img_size + 32)),
                    transforms.RandomCrop((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.tf(img)
        if self.labels is None:
            return x, 0
        return x, int(self.labels[idx])


class DLBackbone:
    """Обёртка timm-backbone с предсказанием вероятности класса 1 (fake)."""

    def __init__(self, cfg: DLConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("Install `torch` and `torchvision` (optional dependency: '[dl]')")
        if not TIMM_AVAILABLE:
            raise ImportError("Install `timm` (optional dependency: '[dl]')")
        self.cfg = cfg
        self.device = cfg.resolve_device()
        self.model = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, num_classes=cfg.num_classes
        ).to(self.device)
        log.info(f"[DL] backbone={cfg.backbone} pretrained={cfg.pretrained} device={self.device}")

    def fit(self, X_paths, y):
        ds = _ImagePathsDataset(X_paths, y, img_size=self.cfg.img_size, train=True)
        loader = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device == "cuda"),
        )
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for ep in range(self.cfg.epochs):
            running = 0.0
            n_seen = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                n_seen += xb.size(0)
            log.info(f"[DL] epoch {ep + 1}/{self.cfg.epochs}  loss={running / max(n_seen, 1):.4f}")
        return self

    def predict_proba(self, X_paths) -> np.ndarray:
        # no_grad контекстом, а не декоратором: @torch.no_grad() на этапе
        # определения класса падал бы при отсутствии torch
        self.model.eval()
        ds = _ImagePathsDataset(X_paths, None, img_size=self.cfg.img_size, train=False)
        loader = DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers
        )
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                probs.append(p)
        return np.concatenate(probs)

    def save(self, path: str | Path):
        torch.save(self.model.state_dict(), str(path))

    def load(self, path: str | Path):
        self.model.load_state_dict(torch.load(str(path), map_location=self.device))
