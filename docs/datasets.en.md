# Datasets and split construction

[Русский](datasets.md) · **English**

Run commands from the repository root. Place the source datasets under `data/raw/` or create directory symlinks there. The public repository contains aggregate results and the code that constructs the splits. Source images are distributed by their respective providers.

| Source | Version used | Role |
| --- | --- | --- |
| [CelebA on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) | Version 2 | Official image partitions with identities separated across splits, real class in A and B |
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | Version 2 | FFHQ real photographs and StyleGAN images. The train, valid and test source directories serve distinct protocol roles |
| [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace/tree/eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c) | Hugging Face revision `eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c` | Real IMDB-WIKI images and SD text-to-image, inpainting and InsightFace manipulation groups |

Select the indicated versions when obtaining the datasets. The general-purpose `scripts/download_data.py` helper uses the provider's default revision.

## A–D · Real-source substitution and processing control

The expected paths are `data/raw/real_fake_140k/real_vs_fake/real-vs-fake/` with `{train,valid,test}/{real,fake}`, and `data/raw/celeba/` with `list_eval_partition.csv` and `img_align_celeba/img_align_celeba/`.

```sh
python -m scripts.prepare_splits data=gan_only seed=42
python -m scripts.make_control_sets a2
python -m scripts.make_control_sets resample --out data/raw/ctrl_resample
python -m scripts.run_negative_control data=celeba_vs_140k pipeline=col2 model=xgboost seed=42 +negctrl.n_repeats=10
```

Build D before A. The A builder copies D's StyleGAN filenames and split membership. A and D each contain 20,000 real and 20,000 synthetic images with 28,000/6,000/6,000 train/validation/test examples. C uses 10,000 real images from 140k valid/real. One half is resized to the 178×218 CelebA geometry. The other half uses 256×256. Both are saved at JPEG quality 95.

## E–F · Source identification and DeepFakeFace

Extract the four groups into `data/raw/dff_extracted/{wiki,text2img,inpainting,insight}`. Keep the provider's original `wiki.zip` in `data/raw/dff/` for the processing-only control. Normalize the four source folders, then the two 140k sources used by the generator-transfer and real-source controls.

```sh
for source in wiki text2img inpainting insight; do
  python scripts/normalize_domain.py --src "data/raw/dff_extracted/$source" \
    --out "data/raw/dm_norm/$source" --size 256 --quality 95 --min-sat 12 --limit 25000
done
base=data/raw/real_fake_140k/real_vs_fake/real-vs-fake/train
python scripts/normalize_domain.py --src "$base/real" --out data/raw/dm_norm/flickr140k \
  --src "$base/fake" --out data/raw/dm_norm/stylegan140k --size 256 --quality 95 --min-sat 12 --limit 25000
python -m scripts.prepare_splits data=dm_only seed=42
python -m scripts.run_logo pipeline=col2 model=xgboost seed=42
```

The normalization uses a central 256-pixel crop, a common JPEG encoder at quality 95 and a symmetric saturation filter. The DeepFakeFace v3 split groups by person across all four sources. The test split contained 2,816 real and 3,151 manipulated images. The E control used 6,000 training and 2,000 test images per real source.

To run F, train the detectors first with `python -m scripts.run_benchmark data=dm_only cv=holdout seed=42`. The 31 handcrafted configurations also include XGBoost runs with `grad1_r256`, `grad2_r256`, `grad4_r256`, `tex4_r256`, `tex5_r256` and `tex6_r256`. Run each with `scripts.run_pipeline` to include them in the control. The command `python -m scripts.processing_control data=dm_only cv=holdout` loads the tracked models and uses the original real images to simulate resize and JPEG processing.

## G · Generator comparisons

The paired v2 protocol starts from 3,000 FFHQ images from 140k test/real. Every original has a Stable Diffusion 1.5 VAE reconstruction, an img2img regeneration and an inpainting variant with the original background. The StyleGAN comparison uses 3,000 independently generated images from 140k test/fake. The preparation code applies a common output path and equalizes JPEG history.

Generation uses CUDA with the Stable Diffusion 1.5 base and inpainting checkpoints. The recorded generation environment used diffusers 0.30.2, transformers 4.44.2 and accelerate 0.33.0. Install these alongside the `dl` extra. The [generator code](../scripts/build_pairs.py) defines the model identifiers, seed derivation, JPEG settings, inpainting mask and img2img strength (0.4).

```sh
python -m scripts.build_pairs generate --out data/raw/pairs_v2 --n 3000 --batch 8
python -m scripts.build_pairs splits --out data/raw/pairs_v2 --version v2
python -m scripts.run_benchmark data=pairs_vae cv=holdout seed=42
python -m scripts.run_pairs_matrix pipeline=col2 model=xgboost seed=42 +pairs.version=v2
```

Originals and their SD variants share a source-group split of 2,113/461/426. The independent StyleGAN images split into 2,118/443/439. The other benchmark configurations are `pairs_img2img`, `pairs_inpaint` and `pairs_stylegan`.
