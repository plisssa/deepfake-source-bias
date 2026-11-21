# Данные и разбиение выборок

**Русский** · [English](datasets.en.md)

Выполняйте команды из корня репозитория. Поместите исходные наборы в `data/raw/` или создайте в этом каталоге символические ссылки. Исходные фотографии доступны у распространителей коллекций.

| Набор данных | Версия в исследовании | Использование |
| --- | --- | --- |
| [CelebA на Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) | 2 | Реальные фотографии в условиях A и B. Официальное разбиение разделяет личности между обучающей, валидационной и тестовой частями |
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | 2 | Реальные фотографии FFHQ и изображения StyleGAN. Для разных условий используются разные исходные каталоги коллекции |
| [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace/tree/eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c) | Версия Hugging Face `eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c` | Реальные фотографии IMDB-WIKI и три группы синтезированных изображений — генерация по тексту, дорисовка и замена лица методом InsightFace |

Для повторения эксперимента скачайте указанные версии. Вспомогательный скрипт `scripts/download_data.py` по умолчанию запрашивает версию, которую на момент скачивания предлагает распространитель.

## Условия A–D. Источник реальных фотографий и контроль обработки

Для коллекции 140k сохраните структуру `data/raw/real_fake_140k/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}`. В `data/raw/celeba/` нужны файл `list_eval_partition.csv` и каталог `img_align_celeba/img_align_celeba/`.

```sh
python -m scripts.prepare_splits data=gan_only seed=42
python -m scripts.make_control_sets a2
python -m scripts.make_control_sets resample --out data/raw/ctrl_resample
python -m scripts.run_negative_control data=celeba_vs_140k pipeline=col2 model=xgboost seed=42 +negctrl.n_repeats=10
```

Сначала подготовьте разбиение D. Скрипт подготовки A использует из него те же файлы StyleGAN в тех же частях выборки. В каждом из условий A и D использованы 20 000 реальных фотографий и 20 000 синтезированных изображений. Из них 28 000 вошли в обучающую выборку, по 6000 — в валидационную и тестовую. Для C берутся 10 000 реальных фотографий из `valid/real`. Половина приводится к геометрии CelebA (178 × 218), другая половина сохраняется в размере 256 × 256. Обе группы сохраняются в JPEG с качеством 95.

## Условия E и F. Разделимость источников и контроль на DeepFakeFace

Распакуйте четыре группы в `data/raw/dff_extracted/{wiki,text2img,inpainting,insight}`. Сохраните исходный `wiki.zip` в `data/raw/dff/` для контроля обработки. Приведите к единому формату четыре группы DeepFakeFace, а затем два источника из 140k.

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

Для всех источников вырезается центральный фрагмент 256 × 256, применяется общий порог насыщенности, используется один кодировщик JPEG с качеством 95. В DeepFakeFace v3 разбиение выполняется по личности сразу для всех четырёх источников. В тестовую часть вошли 2816 реальных и 3151 изменённое изображение. В условии E от каждого реального источника взяты 6000 фотографий для обучения и 2000 для проверки.

Для F обучите основную матрицу командой `python -m scripts.run_benchmark data=dm_only cv=holdout seed=42`. Чтобы получить все 31 конфигурацию с XGBoost, дополнительно запустите `scripts.run_pipeline` для `grad1_r256`, `grad2_r256`, `grad4_r256`, `tex4_r256`, `tex5_r256` и `tex6_r256` с `data=dm_only model=xgboost`. Затем выполните `python -m scripts.processing_control data=dm_only cv=holdout`. Скрипт загрузит обученные модели и оценит их на реальных фотографиях после изменения размера и JPEG-сжатия.

## Условие G. Сравнение генераторов

Парная выборка v2 построена на 3000 реальных фотографиях FFHQ из `test/real`. Для каждой фотографии получены три версии средствами Stable Diffusion 1.5 — реконструкция автокодировщиком, преобразование всего изображения и дорисовка области лица с возвратом исходного фона. Для отдельного сравнения использованы 3000 независимых изображений StyleGAN из `test/fake`. При подготовке классов уравнивается история JPEG-сжатия и применяется общая завершающая обработка.

Для генерации нужны CUDA и веса базовой модели Stable Diffusion 1.5 и её версии для дорисовки. В использованном окружении были установлены diffusers 0.30.2, transformers 4.44.2 и accelerate 0.33.0 дополнительно к зависимостям `dl`. Идентификаторы весов, получение начальных значений генератора случайных чисел, параметры JPEG и маска дорисовки заданы в [коде подготовки пар](../scripts/build_pairs.py). Для преобразования всего кадра параметр `strength` равен 0,4.

```sh
python -m scripts.build_pairs generate --out data/raw/pairs_v2 --n 3000 --batch 8
python -m scripts.build_pairs splits --out data/raw/pairs_v2 --version v2
python -m scripts.run_benchmark data=pairs_vae cv=holdout seed=42
python -m scripts.run_pairs_matrix pipeline=col2 model=xgboost seed=42 +pairs.version=v2
```

Исходная фотография и три её производные попадают в одну часть выборки. Для обучения, валидации и теста выбрано соответственно 2113, 461 и 426 исходников. У независимых изображений StyleGAN распределение составляет 2118, 443 и 439. Кроме `pairs_vae`, доступны конфигурации `pairs_img2img`, `pairs_inpaint` и `pairs_stylegan`.
