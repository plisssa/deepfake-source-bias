"""Fine-tune a pretrained image classifier with optional blur and JPEG augmentation."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from deepfakes.eval.metrics import compute_report, tune_threshold
from deepfakes.io.splits import load_split_csv
from deepfakes.tracking.mlflow_setup import init_mlflow, mlflow_run
from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger

load_project_dotenv()
log = get_logger(__name__)

AUGS = ("none", "blur_jpeg")


# ------------------------------------------------ чистые функции (тестируются)


def crop(img: np.ndarray, size: int, train: bool, rng: np.random.Generator | None) -> np.ndarray:
    """Кроп size×size без интерполяции: случайный при обучении, центральный иначе."""
    h, w = img.shape[:2]
    if h < size or w < size:
        raise ValueError(f"изображение {h}×{w} меньше кропа {size}")
    if train:
        top = int(rng.integers(0, h - size + 1))
        left = int(rng.integers(0, w - size + 1))
    else:
        top, left = (h - size) // 2, (w - size) // 2
    return img[top : top + size, left : left + size]


def blur_jpeg(img: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Аугментация Wang et al. 2020: размытие σ~U[0,3] и JPEG q~U{30..100}, каждое с вер. p."""
    out = img
    if rng.random() < p:
        sigma = float(rng.uniform(0.0, 3.0))
        if sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    if rng.random() < p:
        q = int(rng.integers(30, 101))
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q]
        )
        if ok:
            out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


def variant_name(backbone: str, aug: str) -> str:
    short = backbone.split(".")[0]
    return f"{short}_ft" + ("" if aug == "none" else f"_{aug}")


# ------------------------------------------------------------------ обучение


def _make_dataset(torch, paths, labels, size, train, aug, seed, mean, std):
    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            # seed воркера torch выводится из основного seed и меняется от эпохи к
            # эпохе: аугментации разные в каждой эпохе, но повторяемые при том же
            # seed (в отличие от seed по времени)
            rng = np.random.default_rng((torch.initial_seed(), i)) if train else None
            img = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((size, size, 3), np.uint8)
                valid = 0
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                valid = 1
                if train and aug == "blur_jpeg":
                    img = blur_jpeg(img, rng)
                img = crop(img, size, train, rng)
                if train and rng.random() < 0.5:
                    img = img[:, ::-1]
            x = (img.astype(np.float32) / 255.0 - mean) / std
            return (
                torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                float(labels[i]),
                valid,
            )

    return DS()


def _predict(torch, model, loader, device):
    model.eval()
    scores, valid = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
        for x, _, v in loader:
            s = torch.sigmoid(model(x.to(device, non_blocking=True)).float().squeeze(1))
            scores.append(s.cpu().numpy())
            valid.append(v.numpy())
    return np.concatenate(scores), np.concatenate(valid).astype(bool)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    import timm
    import torch

    configure_logging(cfg.logging.level)
    ccfg = cfg.get("cnn", {}) or {}
    backbone = str(ccfg.get("backbone", "resnet50.tv_in1k"))
    aug = str(ccfg.get("aug", "none"))
    if aug not in AUGS:
        raise ValueError(f"cnn.aug: {AUGS}, получено {aug!r}")
    epochs = int(ccfg.get("epochs", 15))
    patience = int(ccfg.get("patience", 3))
    batch = int(ccfg.get("batch_size", 64))
    lr = float(ccfg.get("lr", 1e-4))
    size = int(ccfg.get("img_size", 224))
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_mlflow(cfg.tracking.tracking_uri, cfg.tracking.experiment_name)
    splits = load_split_csv(Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version)
    limit = int(ccfg.get("limit", 0))  # только для смоук-теста: первые N каждого сплита
    if limit:
        rng_l = np.random.default_rng(cfg.seed)
        for k in splits:
            n = len(splits[k]["X"])
            idx = np.sort(rng_l.choice(n, size=min(limit, n), replace=False))
            splits[k] = {kk: np.asarray(vv)[idx] for kk, vv in splits[k].items()}
        log.warning(f"СМОУК: сплиты урезаны до {limit} — метрики не для отчёта")

    model = timm.create_model(backbone, pretrained=True, num_classes=1).to(device)
    dcfg = timm.data.resolve_model_data_config(model)
    mean = np.array(dcfg["mean"], np.float32)
    std = np.array(dcfg["std"], np.float32)

    def loader(name, train):
        ds = _make_dataset(
            torch, splits[name]["X"], splits[name]["y"], size, train, aug, cfg.seed, mean, std
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch,
            shuffle=train,
            num_workers=workers,
            pin_memory=True,
            drop_last=train,
            persistent_workers=workers > 0,
        )

    tr, va, te = loader("train", True), loader("val", False), loader("test", False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()

    variant = variant_name(backbone, aug)
    run_name = f"{variant}__{cfg.data.name}__finetune__{cfg.cv.name}__seed{cfg.seed}"
    out_dir = Path("reports/runs") / cfg.tracking.experiment_name / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = {
        "pipeline": variant,
        "pipeline_class": "cnn_finetune",
        "model": "finetune",
        "cv": cfg.cv.name,
        "data": cfg.data.name,
        "splits_version": str(cfg.data.splits_version),
        "seed": cfg.seed,
        "kind": "cnn" if not limit else "cnn_smoke",
    }
    with mlflow_run(run_name=run_name, tags=tags):
        mlflow.log_params(
            {
                "backbone": backbone,
                "aug": aug,
                "epochs_max": epochs,
                "patience": patience,
                "batch_size": batch,
                "lr": lr,
                "img_size": size,
                "input": "crop_no_resize",
            }
        )
        best_auc, best_state, bad = -1.0, None, 0
        for ep in range(epochs):
            model.train()
            t0, losses = time.time(), []
            for x, y, v in tr:
                m = v.bool()
                if not m.any():
                    continue
                x, y = x[m].to(device, non_blocking=True), y[m].float().to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                    loss = lossf(model(x).squeeze(1), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach()))
            s_va, v_va = _predict(torch, model, va, device)
            auc_va = roc_auc_score(np.asarray(splits["val"]["y"])[v_va], s_va[v_va])
            sched.step(auc_va)
            mlflow.log_metrics(
                {"train_loss": float(np.mean(losses)), "val_auc_epoch": auc_va}, step=ep
            )
            log.info(
                f"эпоха {ep}: loss={np.mean(losses):.4f} val AUC={auc_va:.4f} ({time.time() - t0:.0f} с)"
            )
            if auc_va > best_auc + 1e-4:
                best_auc, bad = auc_va, 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    log.info(f"ранняя остановка на эпохе {ep}")
                    break
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "best.pt")

        # порог — по F1 на val, как в бенчмарке; метрики — тем же compute_report
        s_va, v_va = _predict(torch, model, va, device)
        y_va = np.asarray(splits["val"]["y"])[v_va]
        thr = tune_threshold(y_va, s_va[v_va], objective="f1")
        for name, ld in (("val", va), ("test", te)):
            s, v = _predict(torch, model, ld, device)
            y = np.asarray(splits[name]["y"])[v]
            rep = compute_report(y, s[v], threshold=thr)
            mlflow.log_metrics(rep.flatten_for_mlflow(prefix=f"{name}_"))
            if name == "test":
                from scripts.run_pipeline import save_test_predictions

                p = save_test_predictions(out_dir, splits["test"]["X"], v, y, s[v], thr)
                mlflow.log_artifact(str(p))
                log.info(f"TEST  AUC={rep.roc_auc:.4f}  F1={rep.f1:.4f}  EER={rep.eer:.4f}")
        mlflow.log_param("threshold_chosen", float(thr))
        mlflow.log_metric("best_val_auc", best_auc)


if __name__ == "__main__":
    main()
