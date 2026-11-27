# Deepfake Source Bias

[![Tests](https://github.com/plisssa/deepfake-source-bias/actions/workflows/tests.yml/badge.svg)](https://github.com/plisssa/deepfake-source-bias/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-086b69.svg)](LICENSE)

[Русский](README.md) · **English**

An evaluation of how image provenance and preprocessing affect face deepfake detectors. The benchmark compares **25 feature extractors paired with three classifiers** and includes separate experiments with CLIP, DINOv2, ResNet-50 and a ViT-Tiny hybrid.

[Dataset preparation](docs/datasets.en.md) · [Models and training](docs/models.en.md) · [Results](docs/results.en.md)

![Design of the main comparison. The same StyleGAN images are compared with real CelebA photographs (median ROC-AUC 0.9996) and real FFHQ photographs (0.7837). In the control, two differently processed groups of real FFHQ photographs are separated with a median ROC-AUC of 0.9844.](docs/assets/design.en.png)

The **same StyleGAN test images** produced markedly different scores depending on the real comparison set. The median test ROC-AUC was **0.9996** against CelebA and **0.7837** against FFHQ. A control that contains no synthetic images at all, only real FFHQ photographs processed in two different ways, reached **0.9844**. The near-perfect score in condition A therefore largely reflects the source and processing of the real photographs rather than traces of synthesis.

![Test ROC-AUC distributions over 75 configurations for conditions A, D, C and four generators on FFHQ.](docs/assets/benchmark.en.png)

Each row summarizes variation across 75 detector configurations at seed 42.

## Transfer across generators

![Train–test matrix for four generators. Median ROC-AUC is 0.538 for VAE to StyleGAN and 0.624 in the reverse direction.](docs/assets/transfer.en.png)

Each cell shows the median test ROC-AUC across ten feature sets with XGBoost. For models trained on VAE reconstructions, the median is 0.998 on img2img and 0.538 on StyleGAN. Rows indicate training, columns indicate testing.

The diagonal is higher than the medians in the previous figure because this matrix covers only the ten XGBoost feature sets. The previous figure takes the median over all 75 configurations, including logistic regression and random forest.

[Plots for preprocessing and neural detectors](docs/results.en.md).

## Quick start

Recalculate the summaries from the published per-configuration metrics. Python 3.10+ is sufficient. The command runs offline and uses the standard library.

```sh
git clone https://github.com/plisssa/deepfake-source-bias.git
cd deepfake-source-bias
python3 analysis/summarize.py --lang en
```

## Train a detector

Install the package from the checkout and prepare the FFHQ/StyleGAN corpus as described in the [dataset guide](docs/datasets.en.md). This example trains an XGBoost classifier on block-level RGB statistics for condition D.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m scripts.prepare_splits data=gan_only seed=42
python -m scripts.run_pipeline data=gan_only pipeline=col2 model=xgboost cv=holdout seed=42
```

To run the test suite on synthetic fixtures, use `python -m pip install -e '.[dev]'` and `python -m pytest -q`. The figures are produced by `analysis/plot_design.py`, `analysis/plot_benchmark.py` and `analysis/plot_experiments.py` with `--lang en`.

## Neural baselines

The rows below come from completed cluster runs. The FFHQ/StyleGAN test has 6,000 images. The DeepFakeFace test has 5,967.

| Model | FFHQ / StyleGAN AUC | DeepFakeFace AUC |
| --- | ---: | ---: |
| Frozen CLIP ViT-L/14 + logistic regression | 0.9990 | 0.9998 |
| Frozen DINOv2 ViT-B/14 + logistic regression | 0.9940 | 0.9564 |
| Fine-tuned ResNet-50 | > 0.9999 | 0.9999 |

In the processing control, CLIP separates two preparations of the same real photographs with ROC-AUC 0.9993, and DINOv2 with 0.8582. The [model catalog](docs/models.en.md) covers XGBoost heads, blur/JPEG augmentation and both ViT-Tiny preprocessing variants.

## Code and data

- [`deepfakes/`](deepfakes/) contains the feature extractors, estimators and evaluation routines. [`configs/`](configs/) holds the Hydra configurations.
- [`scripts/`](scripts/) contains the data preparation, training, control and transfer experiments. The [cluster environment](requirements-cluster.lock.txt) records the versions used for the reported results.
- [`results/`](results/) contains aggregate scores for each model or configuration. The [results guide](docs/results.en.md) maps the seven experimental conditions to the tables and explains their scope.
- [`analysis/`](analysis/) recomputes the summary statistics and draws the figures.

## Paper

The accompanying paper is under review. A link will be added here once it is published.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff); GitHub uses it for the "Cite this repository" menu.

```bibtex
@software{puzyreva2026deepfakesourcebias,
  author  = {Puzyreva, Elizaveta},
  title   = {Deepfake Source Bias},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/plisssa/deepfake-source-bias}
}
```

The software is distributed under the [MIT license](LICENSE). The [dataset guide](docs/datasets.en.md) links to the original image collections.
