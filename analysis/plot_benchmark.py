"""Построение графика для README по тестовой ROC-AUC отдельных конфигураций."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from summarize import benchmark, language_argument, number

ROOT = Path(__file__).resolve().parents[1]
INK, SUBTLE, GRID = "#162a36", "#415665", "#d4dcdf"
TEAL, OCHRE = "#086b69", "#9c5129"


def panel(ax, values, protocols, labels, color, title, lang):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=15)
    for row, protocol in enumerate(protocols):
        scores = np.array(list(values[protocol].values()))
        low, q1, mid, q3, high = np.quantile(scores, [0, .25, .5, .75, 1])
        ax.hlines(row, low, high, color=SUBTLE, linewidth=1.1)
        ax.hlines(row, q1, q3, color=color, linewidth=8, alpha=.55)
        ax.plot(mid, row, marker="o", color=color, markersize=7, markeredgecolor="white")
        ax.text(1.02, row, number(mid, 4, lang), va="center", color=INK, fontsize=9, clip_on=False)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=10, color=INK)
    ax.set_ylim(len(labels)-.45, -.58)
    ax.set_xlim(.42, 1.01)
    ax.set_xticks([.5, .6, .7, .8, .9, 1.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: number(value, 1, lang)))
    ax.grid(axis="x", color=GRID, lw=.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=6, colors=SUBTLE)
    xlabel = "Test ROC-AUC" if lang == "en" else "ROC-AUC на тестовой выборке"
    ax.set_xlabel(xlabel, color=SUBTLE, labelpad=9, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    lang = language_argument()
    if lang == "en":
        processing = "FFHQ processing control"
        generators = ["VAE reconstruction", "Image-to-image", "Inpainting", "StyleGAN"]
        titles = ["Real-image source", "Generator comparisons on FFHQ"]
        legend = "75 configurations per row. The dot marks the median, the band the middle 50%, and the line the full range."
        filename = "benchmark.en.png"
    else:
        processing = "Обработка FFHQ"
        generators = ["Реконструкция VAE", "Преобразование кадра", "Дорисовка", "StyleGAN"]
        titles = ["Источник реальных фотографий", "Сравнение генераторов на FFHQ"]
        legend = "В каждой строке 75 конфигураций. Точка — медиана, полоса — центральные 50 % значений, линия — весь диапазон."
        filename = "benchmark.png"
    values = benchmark()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), facecolor="white")
    fig.subplots_adjust(left=.146, right=.94, top=.81, bottom=.31, wspace=.94)
    panel(axes[0], values, ["A", "C", "D"],
          ["CelebA / StyleGAN", processing, "FFHQ / StyleGAN"], TEAL, titles[0], lang)
    panel(axes[1], values, ["G-vae", "G-img2img", "G-inpaint", "G-stylegan"],
          generators, OCHRE, titles[1], lang)
    fig.text(.5, .075, legend, ha="center", color=SUBTLE, fontsize=10)
    fig.savefig(ROOT / "docs/assets" / filename, dpi=160, facecolor="white")
    plt.close(fig)
