"""Пересчёт сводных показателей по значениям ROC-AUC отдельных конфигураций."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import correlation, median

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONDITIONS = {
    "A": "CelebA и StyleGAN",
    "C": "Контроль обработки FFHQ",
    "D": "FFHQ и StyleGAN",
    "F-test": "Тест DeepFakeFace",
    "G-vae": "Реконструкция VAE",
    "G-img2img": "Преобразование кадра",
    "G-inpaint": "Дорисовка",
    "G-stylegan": "Сравнение со StyleGAN",
}


EN_CONDITIONS = {
    "A": "CelebA and StyleGAN",
    "C": "FFHQ processing control",
    "D": "FFHQ and StyleGAN",
    "F-test": "DeepFakeFace test",
    "G-vae": "VAE reconstruction",
    "G-img2img": "Image-to-image",
    "G-inpaint": "Inpainting",
    "G-stylegan": "StyleGAN comparison",
}


def language_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=("ru", "en"), default="ru", help="Язык / language")
    return parser.parse_args().lang


def rows(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def benchmark():
    conditions = defaultdict(dict)
    for row in rows("feature-benchmark.csv"):
        group = conditions[row["condition"]]
        key = row["feature"], row["classifier"]
        if key in group:
            raise ValueError(f"Повтор конфигурации в {row['condition']}: {key}")
        group[key] = float(row["roc_auc"])
    if set(conditions) != set(CONDITIONS) or len(conditions["A"]) != 75:
        raise ValueError("В основной матрице отсутствуют условия или конфигурации")
    for name, group in conditions.items():
        if group.keys() != conditions["A"].keys():
            raise ValueError(f"Набор конфигураций расходится с условием A в {name}")
    return conditions


def number(value, places, lang="ru"):
    text = f"{value:.{places}f}"
    return text.replace(".", ",") if lang == "ru" else text


def main(lang="ru"):
    if lang == "en":
        labels = EN_CONDITIONS
        condition, comparison, metric = "Condition", "Comparison", "Median test ROC-AUC"
        count, controls = "configurations", "Control experiments"
        geometry, source, processing = "Image geometry / file size", "Source ROC-AUC range", "Test vs. processing control"
    else:
        labels = CONDITIONS
        condition, comparison, metric = "Условие", "Сравнение", "Медианная тестовая ROC-AUC"
        count, controls = "конфигураций", "Контрольные эксперименты"
        geometry, source, processing = "Размер кадра / объём файла", "Диапазон ROC-AUC по источникам", "Тест и контроль обработки"

    print(f"{condition:<10} {comparison:<27} {metric}")
    for key, values in benchmark().items():
        mid = number(median(values.values()), 6, lang)
        print(f"{key:<10} {labels[key]:<27} {mid}   ({len(values)} {count})")

    print(f"\n{controls}")
    basic = rows("file-controls.csv")[0]
    sizes = [number(float(basic[k]), 3, lang) for k in ("geometry_auc", "filesize_auc")]
    print(f"B          {geometry:<30} {' / '.join(sizes)}")
    sources = [float(r["roc_auc"]) for r in rows("source-control.csv")]
    print(f"E          {source:<30} {number(min(sources), 4, lang)}–{number(max(sources), 4, lang)}")
    control = [r for r in rows("processing-control.csv") if r["head"] == "xgboost"
               and r["feature"] not in {"clip_l14_lp", "dinov2_b14_lp"}]
    coefficient = correlation([float(r["test_auc"]) for r in control],
                              [float(r["bicubic_jpeg_auc"]) for r in control])
    print(f"F          {processing:<30} r = {number(coefficient, 6, lang)}   (n = {len(control)})")


if __name__ == "__main__":
    main(language_argument())
