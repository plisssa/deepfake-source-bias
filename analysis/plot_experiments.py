"""Графики кластерных экспериментов по открытым сводным таблицам."""

from collections import defaultdict
from statistics import correlation, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np

from plot_benchmark import GRID, INK, OCHRE, ROOT, SUBTLE, TEAL
from summarize import language_argument, number, rows


def transfer(lang):
    en = lang == "en"
    keys = ["vae", "img2img", "inpaint", "stylegan"]
    names = ["VAE", "Img2img", "Inpainting" if en else "Дорисовка", "StyleGAN"]
    groups = defaultdict(list)
    for record in rows("generator-transfer.csv"):
        groups[record["train_generator"], record["test_generator"]].append(float(record["roc_auc"]))
    matrix = [[median(groups[train, test]) for test in keys] for train in keys]
    cmap = LinearSegmentedColormap.from_list("transfer", [OCHRE, "#faf7f0", TEAL])
    fig, ax = plt.subplots(figsize=(6.7, 4.3), layout="constrained")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(matrix):
        for j, score in enumerate(row):
            ax.text(j, i, number(score, 3, lang), ha="center", va="center", fontsize=13,
                    color="white" if score > .81 else INK)
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.set_xlabel("Test generator" if en else "Генератор в тестовой выборке")
    ax.set_ylabel("Training generator" if en else "Генератор в обучающей выборке")
    ax.set_title("Transfer across generators" if en else "Перенос между генераторами")
    bar = fig.colorbar(im, ax=ax, pad=.04, ticks=[0, .25, .5, .75, 1],
                       format=FuncFormatter(lambda x, _: number(x, 2, lang)))
    bar.set_label("Median ROC-AUC" if en else "Медианная ROC-AUC")
    return fig


def processing(lang):
    en = lang == "en"
    features = [r for r in rows("processing-control.csv") if r["head"] == "xgboost"
                and r["feature"] not in {"clip_l14_lp", "dinov2_b14_lp"}]
    test = [float(r["test_auc"]) for r in features]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True, layout="constrained")
    panels = [("jpeg_auc", "JPEG", OCHRE),
              ("bicubic_jpeg_auc", "Bicubic resize + JPEG" if en else "Бикубическое масштабирование + JPEG", TEAL)]
    for ax, (field, title, color) in zip(axes, panels):
        control = [float(r[field]) for r in features]
        ax.plot([.35, 1], [.35, 1], ls="--", lw=.8, color=SUBTLE)
        ax.scatter(control, test, s=40, color=color, alpha=.75, edgecolor="white", linewidth=.5)
        ax.set(aspect="equal", xlim=(.35, 1.024), ylim=(.35, 1.024))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Control ROC-AUC" if en else "ROC-AUC в контроле")
        ax.text(.96, .07, "r = " + number(correlation(control, test), 3, lang),
                transform=ax.transAxes, ha="right", fontsize=12, color=color)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(FuncFormatter(lambda x, _: number(x, 1, lang)))
        ax.grid(color=GRID, lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test ROC-AUC on DeepFakeFace" if en else "Тестовая ROC-AUC на DeepFakeFace")
    return fig


def neural(lang):
    en = lang == "en"
    variants = [
        ("clip_l14_lp", "logreg", "CLIP ViT-L/14 · " + ("logistic regression" if en else "лог. регрессия")),
        ("clip_l14_lp", "xgboost", "CLIP ViT-L/14 · XGBoost"),
        ("dinov2_b14_lp", "logreg", "DINOv2 ViT-B/14 · " + ("logistic regression" if en else "лог. регрессия")),
        ("dinov2_b14_lp", "xgboost", "DINOv2 ViT-B/14 · XGBoost"),
        ("resnet50_ft", "finetune", "ResNet-50"),
        ("resnet50_ft_blur_jpeg", "finetune", "ResNet-50 · " + ("blur + JPEG" if en else "размытие + JPEG")),
        ("rob_deep_fusion", "xgboost", "ViT-Tiny · " + ("resize" if en else "масштабирование")),
        ("rob_deep_fusion_fixed", "xgboost", "ViT-Tiny · " + ("center crop" if en else "центральный фрагмент")),
    ]
    scores = {(r["variant"], r["head"], r["condition"]): float(r["roc_auc"])
              for r in rows("neural-baselines.csv")}
    fig, ax = plt.subplots(figsize=(9.3, 4.7), layout="constrained")
    for idx, (variant, head, _) in enumerate(variants):
        d = scores[variant, head, "D"]
        f = scores.get((variant, head, "F-test"))
        if f is not None:
            ax.plot([d, f], [idx, idx], color=GRID, lw=2.2, zorder=1)
            ax.scatter(f, idx, marker="D", s=43, color=OCHRE, label="DeepFakeFace" if idx == 0 else None)
        ax.scatter(d, idx, marker="o", s=83, facecolors="none", edgecolors=TEAL,
                   linewidth=1.6, label="FFHQ / StyleGAN" if idx == 0 else None)
    ax.set_yticks(range(len(variants)), [label for _, _, label in variants])
    ax.invert_yaxis()
    ax.set_xlim(.887, 1.005)
    ax.set_xticks(np.arange(.90, 1.01, .02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: number(x, 2, lang)))
    ax.set_xlabel("Test ROC-AUC" if en else "Тестовая ROC-AUC")
    ax.grid(axis="x", color=GRID, lw=.65)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Neural detectors on two test sets" if en else "Нейросетевые детекторы на двух выборках")
    return fig


if __name__ == "__main__":
    lang = language_argument()
    suffix = ".en.png" if lang == "en" else ".png"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": SUBTLE,
                         "xtick.color": SUBTLE, "ytick.color": SUBTLE})
    for name, plot in [("transfer", transfer), ("processing", processing), ("neural", neural)]:
        figure = plot(lang)
        figure.savefig(ROOT / "docs/assets" / (name + suffix), dpi=160)
        plt.close(figure)
