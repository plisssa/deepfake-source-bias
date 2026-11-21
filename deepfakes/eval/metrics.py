"""Единый модуль метрик: classification + calibration + bootstrap CI + EER."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationReport:
    """Полный отчёт по бинарной классификации."""

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier: float
    eer: float
    threshold: float
    confusion_matrix: list[list[int]]
    n: int
    n_pos: int
    n_neg: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def flatten_for_mlflow(self, prefix: str = "") -> dict[str, float]:
        """Развернуть отчёт в плоский словарь метрик для mlflow.log_metrics."""
        skip = {"confusion_matrix"}
        out = {}
        for k, v in self.to_dict().items():
            if k in skip:
                continue
            if isinstance(v, (int, float)):
                out[f"{prefix}{k}"] = float(v)
        # confusion matrix как отдельные ячейки
        cm = self.confusion_matrix
        out[f"{prefix}cm_tn"] = float(cm[0][0])
        out[f"{prefix}cm_fp"] = float(cm[0][1])
        out[f"{prefix}cm_fn"] = float(cm[1][0])
        out[f"{prefix}cm_tp"] = float(cm[1][1])
        return out


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate (EER) и пороговая точка.

    При вырожденном y_true (один класс) ROC не определён — возвращаем nan
    вместо мусора из предупреждений sklearn."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan"), 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    diffs = np.abs(fpr - fnr)
    idx = int(np.argmin(diffs))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    # в новых sklearn thr[0] == inf — как порог он бессмыслен
    t = thr[idx]
    if not np.isfinite(t):
        t = 1.0
    return float(eer), float(t)


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Площадь под precision-recall кривой."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_score))


def compute_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationReport:
    """Полный отчёт по бинарной классификации.

    Args:
        y_true: метки {0, 1}.
        y_score: предсказанные вероятности класса 1.
        threshold: порог для бинаризации.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    eer, _ = compute_eer(y_true, y_score)

    return ClassificationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        roc_auc=float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        pr_auc=pr_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
        log_loss=float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7), labels=[0, 1])),
        brier=float(brier_score_loss(y_true, y_score)),
        eer=eer,
        threshold=float(threshold),
        confusion_matrix=cm,
        n=len(y_true),
        n_pos=int((y_true == 1).sum()),
        n_neg=int((y_true == 0).sum()),
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap доверительный интервал для произвольной метрики.

    Returns:
        (point_estimate, lower, upper)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    scores = np.asarray(scores)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(scores, alpha))
    hi = float(np.quantile(scores, 1.0 - alpha))
    point = float(metric_fn(y_true, y_score))
    return point, lo, hi


# ---------- threshold tuning ----------


def tune_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    objective: str = "f1",
    target_fpr: float | None = None,
) -> float:
    """Подобрать порог по нужному критерию.

    Args:
        objective: 'f1' | 'youden' | 'eer' | 'target_fpr'.
        target_fpr: если objective='target_fpr', максимизировать TPR при FPR <= target_fpr.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5  # порог не определён без обоих классов
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # sklearn>=1.3: thr[0] == inf — исключаем нефинитные пороги
    thr = np.where(np.isfinite(thr), thr, 1.0)

    if objective == "youden":
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])
    if objective == "eer":
        _, t = compute_eer(y_true, y_score)
        return float(t)
    if objective == "target_fpr":
        if target_fpr is None:
            raise ValueError("target_fpr must be set for objective='target_fpr'")
        ok = fpr <= target_fpr
        if not ok.any():
            return float(thr[int(np.argmax(tpr))])
        best_idx = int(np.argmax(tpr * ok))
        return float(thr[best_idx])
    if objective == "f1":
        ts = np.unique(np.concatenate([[0.0, 1.0], thr]))
        best_t, best_f1 = 0.5, -1.0
        for t in ts:
            preds = (y_score >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return best_t

    raise ValueError(f"Unknown objective: {objective}")


# ---------- plots ----------


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    out_path: str | Path,
    title: str = "Confusion matrix",
    class_names: tuple[str, str] = ("real", "fake"),
) -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: str | Path,
    title: str = "ROC curve",
) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: str | Path,
    title: str = "Precision-Recall curve",
) -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(r, p, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: str | Path,
    n_bins: int = 10,
    title: str = "Calibration",
) -> Path:
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(prob_pred, prob_true, "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="best")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
