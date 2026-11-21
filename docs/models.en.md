# Models and training

[Русский](models.md) · **English**

## Feature benchmark

Each primary comparison crosses 25 feature pipelines with three classifier heads. The primary grid uses fixed model settings. Decision thresholds are selected on the validation split. Reported ROC-AUC is measured on the held-out test split at seed 42.

| Features | Pipelines |
| --- | --- |
| Color | `col1–3` |
| Frequency | `freq1–5` |
| Gradients | `grad1–4` |
| Texture | `tex1–6` |
| Noise and compression | `noise1–3`, `ela` |
| Feature combinations | `ml6_hybrid`, `ml7_freq_lbp`, `rob1_fusion` |

| Head | Primary configuration |
| --- | --- |
| Logistic regression | StandardScaler, L2, C=1, lbfgs, 2,000 maximum iterations, balanced classes |
| Random forest | 300 trees, unrestricted depth, minimum leaf size 2, square-root feature sampling, balanced classes |
| XGBoost | 300 estimators, maximum depth 6, learning rate 0.05, row and column sampling 0.8, histogram tree method |

[Pipeline settings](../configs/pipeline/) determine image sizes and feature-specific transforms. [Classifier settings](../configs/model/) and the [estimator implementation](../deepfakes/models/classical.py) specify defaults. Additional estimator implementations in the package are available for experimentation. The primary grid uses the three heads above.

## Neural experiments

The catalog was checked against model parameters, checkpoint identifiers, experiment scope and test-set sizes in the cluster tracking records. There are 14 full holdout comparisons in [neural-baselines.csv](../results/neural-baselines.csv). The comparison uses the 6,000-image and 5,967-image test sets. A preliminary ResNet-50 check used 300 test images and is excluded from these comparisons.

| Backbone | Training | Evaluations |
| --- | --- | --- |
| CLIP ViT-L/14, `vit_large_patch14_clip_224.openai` | Frozen encoder, central crop of 224, checkpoint normalization, batch size 64. Logistic regression and XGBoost heads | Both test domains, both heads and preprocessing controls. Additional robustness and generator-transfer runs used the logistic head |
| DINOv2 ViT-B/14, `vit_base_patch14_dinov2.lvd142m` | Frozen encoder, 224-pixel crop and positional embedding interpolation, checkpoint normalization, batch size 64. Logistic regression and XGBoost heads | Both test domains, both heads and preprocessing controls and a generator-transfer run with the logistic head |
| ResNet-50, `resnet50.tv_in1k` | End-to-end fine-tuning, 224-pixel crop, Adam at 0.0001, batch size 64, maximum 15 epochs, early stopping patience 3 | Both domains with two training variants, plain and blur + JPEG |
| ViT-Tiny, `vit_tiny_patch16_224` | Frozen embedding with frequency, LBP and Gabor descriptors, PCA to 128 components, then XGBoost | FFHQ / StyleGAN. Resize with ImageNet normalization gave 0.8979. A center crop with checkpoint normalization gave 0.9164 |

The 3 classifiers and the 14 neural comparisons address different questions. The paired-image **generators** are Stable Diffusion 1.5 VAE reconstruction, img2img and inpainting. The StyleGAN arm uses existing synthetic images from the 140k collection. InsightFace is a synthesis method represented within DeepFakeFace.

## Run an experiment

With the [data](datasets.en.md) prepared, run the full primary grid or a single model. Neural backbones require the `dl` package extra and pretrained checkpoints.

```sh
python -m scripts.run_benchmark data=gan_only cv=holdout seed=42
python -m pip install -e '.[dl]'
python -m scripts.run_pipeline data=dm_only pipeline=clip_l14_lp model=logreg seed=42
python -m scripts.run_pipeline data=dm_only pipeline=dinov2_b14_lp model=xgboost seed=42
python -m scripts.train_cnn data=gan_only cv=holdout +cnn.aug=blur_jpeg seed=42
python -m scripts.run_pipeline data=gan_only pipeline=rob_deep_fusion_fixed model=xgboost seed=42
```

The recorded cluster stack used Python 3.12.4, NumPy 1.26.4, scikit-learn 1.9.1, OpenCV 5.0.0.93, XGBoost 3.4.1 and PyTorch 2.4.0 with CUDA 12.4. [Environment snapshot](../requirements-cluster.lock.txt).
