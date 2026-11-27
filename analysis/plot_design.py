"""Схема основного эксперимента: одни и те же синтезированные лица против разных реальных."""

from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_benchmark import GRID, INK, OCHRE, ROOT, SUBTLE, TEAL
from summarize import benchmark, language_argument, number

TEXT = {
    "ru": {
        "headers": ("Класс 1", "Класс 0", "Медианная ROC-AUC"),
        "rows": [
            ("A", "StyleGAN\nиз 140k Real and Fake Faces", "CelebA\nреальные фотографии"),
            ("D", "Те же изображения StyleGAN\nи то же разбиение", "FFHQ\nреальные фотографии"),
            ("C", "FFHQ, приведённые\nк 178 × 218, JPEG 95", "FFHQ, 256 × 256\nJPEG 95"),
        ],
        "control": "контроль:\nфейков нет",
        "note": "Медиана по 75 детекторам: 25 наборов признаков × 3 классификатора, тестовая выборка.",
        "file": "design.png",
    },
    "en": {
        "headers": ("Class 1", "Class 0", "Median ROC-AUC"),
        "rows": [
            ("A", "StyleGAN\nfrom 140k Real and Fake Faces", "CelebA\nreal photographs"),
            ("D", "The same StyleGAN images\nand the same split", "FFHQ\nreal photographs"),
            ("C", "FFHQ resized\nto 178 × 218, JPEG 95", "FFHQ, 256 × 256\nJPEG 95"),
        ],
        "control": "control:\nno fakes",
        "note": "Median over 75 detectors: 25 feature sets × 3 classifiers, held-out test split.",
        "file": "design.en.png",
    },
}


def box(ax, x, y, w, h, text, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.018",
                                linewidth=1.2, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=INK,
            linespacing=1.35)


def main(lang):
    t = TEXT[lang]
    values = benchmark()
    fig, ax = plt.subplots(figsize=(10.5, 4.1), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, mid, right, w, h = .08, .43, .82, .3, .17
    for x, header in zip((left + w / 2, mid + w / 2, right + .08), t["headers"]):
        ax.text(x, .93, header, ha="center", va="center", fontsize=10.5, color=SUBTLE,
                fontweight="bold")

    for i, (key, fake, real) in enumerate(t["rows"]):
        y = .66 - i * .27
        control = key == "C"
        ax.text(.025, y + h / 2, key, ha="center", va="center", fontsize=13,
                fontweight="bold", color=SUBTLE if control else INK)
        box(ax, left, y, w, h, fake, GRID if control else OCHRE,
            "#f6f8f8" if control else "#f7ece6")
        box(ax, mid, y, w, h, real, TEAL if not control else GRID,
            "#e6f0ef" if not control else "#f6f8f8")
        ax.text(left + w + (mid - left - w) / 2, y + h / 2, "vs", ha="center", va="center",
                fontsize=10, color=SUBTLE, style="italic")
        ax.add_patch(FancyArrowPatch((mid + w + .012, y + h / 2), (right - .012, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=12, color=SUBTLE, lw=1))
        score = number(median(values[key].values()), 4, lang)
        ax.text(right + .08, y + h / 2 + (.025 if control else 0), score, ha="center",
                va="center", fontsize=17, fontweight="bold", color=SUBTLE if control else INK)
        if control:
            ax.text(right + .08, y + h / 2 - .06, t["control"], ha="center", va="center",
                    fontsize=8.5, color=SUBTLE, linespacing=1.2)

    ax.text(.5, .025, t["note"], ha="center", va="center", fontsize=9.5, color=SUBTLE)
    fig.savefig(ROOT / "docs/assets" / t["file"], dpi=160, facecolor="white",
                bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


if __name__ == "__main__":
    main(language_argument())
