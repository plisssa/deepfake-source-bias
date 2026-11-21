"""Собственный парный набор с общей предобработкой.

Зачем. На DeepFakeFace модели различают не подделку, а обработку датасета
(RESULTS §18): фейки сжаты авторами до 512×512 JPEG q75, реальные — нет. Здесь
реальное и поддельное различаются ТОЛЬКО шагом генерации.

Версия v2 (29.09.2026). В v1 все классы проходили общий хвост, но у реальных
до него уже было одно сжатие — JPEG q95 4:2:0 от авторов 140k, на той же сетке
8×8, а у сгенерированных пикселей нет. Финальное q95 почти не меняло реальные
кадры (повторное сжатие по той же сетке идемпотентно) и добавляло свежие блоки
фейкам. Одна «блочность» давала AUC 0.94. В v2 у КАЖДОГО пикселя
каждого класса перед хвостом ровно одно сжатие q95 4:2:0 на той же сетке:

    real      x = исходник 140k (сжат авторами q95 4:2:0)
    SD-фейк   x = JPEG_q95_420( BOX_256( G( Lanczos_512(исходник) ) ) )
    inpaint   x = то же для лица, фон — исходные пиксели, вставка на 256
    stylegan  x = фейк 140k (сжат авторами q95 4:2:0)
    хвост     JPEG q95 4:4:4( BOX_256( Lanczos_512(x) ) ) — общий для всех

Реальный кадр и его фейки сделаны из ТОГО ЖЕ исходника; размер, ресэмплинг,
история и параметры сжатия у классов совпадают по построению.

Источник — FFHQ из набора 140k, папки test/real и test/fake: в обучении не
участвовали ни разу (сплиты gan_only берут только train/). Лица в FFHQ
выровнены, поэтому одна эллиптическая маска накрывает лицо на всех кадрах.

Генераторы:
    real      тождество (реальный класс)
    vae       только кодировщик-декодер SD 1.5 (следы декодера без диффузии)
    img2img   SD 1.5, strength 0.4 — регенерация всего кадра
    inpaint   SD 1.5 Inpainting по маске лица, фон вставлен обратно из оригинала
    stylegan  фейки 140k test/fake (StyleGAN, обучен на FFHQ) через тот же путь

Сплит — по исходному кадру (хэш имени): реальный кадр и все его подделки
всегда в одном сплите, и разбиение одинаково во всех вариантах набора.

Остаточный канал — свойство генерации, а не подготовки: сгенерированные
пиксели содержат детали выше полосы апскейла 256→512. Меряется контролями
(пол по весу файла и блочности в run_pairs_matrix).

    python -m scripts.build_pairs generate --out DIR --n 3000 --part 0 --nparts 2
    python -m scripts.build_pairs splits --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from deepfakes.utils.env import load_project_dotenv

# HF_HOME из .env: веса SD лежат на scratch, compute-узлы офлайн
load_project_dotenv()

REPO = Path(__file__).resolve().parents[1]
SRC_140K = REPO / "data/raw/real_fake_140k/real_vs_fake/real-vs-fake/test"

GENERATORS = ("real", "vae", "img2img", "inpaint", "stylegan")
SD_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_INPAINT = "stable-diffusion-v1-5/stable-diffusion-inpainting"
PROMPT = "a high quality photo of a person's face"
STEPS = 30
GUIDANCE = 7.5
I2I_STRENGTH = 0.4
# эллипс лица в долях кадра; подобран по выровненным кадрам FFHQ
FACE_ELLIPSE = (0.50, 0.56, 0.29, 0.36)  # cx, cy, полуось x, полуось y
FEATHER = 3  # px на 256: мягкий шов при вставке фона обратно
# Сжатие, которое авторы 140k применили к реальным и StyleGAN-кадрам (таблицы
# квантования и субдискретизация проверены по 500 файлам каждого класса; это
# умолчание PIL для quality=95). SD-фейки получают его же — см. docstring.
SRC_JPEG = dict(quality=95, subsampling=2)
SPLIT_EDGES = (70, 85)  # train < 70 ≤ val < 85 ≤ test (проценты)


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def split_of(stem: str) -> str:
    """Сплит исходного кадра: детерминирован и не зависит от состава набора."""
    b = _h(f"pairs-split:{stem}") % 100
    return "train" if b < SPLIT_EDGES[0] else "val" if b < SPLIT_EDGES[1] else "test"


def select_sources(files: list[Path], n: int) -> list[Path]:
    """n исходников с наименьшим хэшем имени — детерминированно, без rng."""
    return sorted(files, key=lambda p: _h(f"pairs-pick:{p.name}"))[:n]


def seed_of(gen: str, stem: str) -> int:
    return _h(f"{gen}:{stem}") % (2**31)


def up512(im: Image.Image) -> Image.Image:
    return im.convert("RGB").resize((512, 512), Image.LANCZOS)


def finish(im512: Image.Image) -> bytes:
    """Общий хвост для всех классов: BOX до 256 и JPEG q95 без субдискретизации."""
    assert im512.size == (512, 512), im512.size
    buf = io.BytesIO()
    im512.convert("RGB").resize((256, 256), Image.BOX).save(
        buf, format="JPEG", quality=95, subsampling=0
    )
    return buf.getvalue()


def source_history(im256: Image.Image) -> Image.Image:
    """Одно сжатие как у исходников 140k: JPEG q95 4:2:0 и обратно."""
    assert im256.size == (256, 256), im256.size
    buf = io.BytesIO()
    im256.convert("RGB").save(buf, format="JPEG", **SRC_JPEG)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def as_source(gen512: Image.Image) -> Image.Image:
    """Выход генератора → «исходник»: BOX до 256 и то же сжатие, что у 140k."""
    return source_history(gen512.convert("RGB").resize((256, 256), Image.BOX))


def face_mask(size: int = 512) -> tuple[Image.Image, Image.Image]:
    """(жёсткая маска для инпейнтинга, размытая — для вставки фона обратно)."""
    cx, cy, ax, ay = FACE_ELLIPSE
    hard = Image.new("L", (size, size), 0)
    ImageDraw.Draw(hard).ellipse(
        [(cx - ax) * size, (cy - ay) * size, (cx + ax) * size, (cy + ay) * size], fill=255
    )
    return hard, hard.filter(ImageFilter.GaussianBlur(FEATHER * size / 256))


def composite(gen: Image.Image, orig: Image.Image, soft: Image.Image) -> Image.Image:
    """Внутри маски — сгенерированное, снаружи — исходные пиксели без изменений."""
    return Image.composite(gen.convert("RGB"), orig.convert("RGB"), soft)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _is_degenerate(im: Image.Image) -> bool:
    """Чёрный или однотонный выход (NaN в fp16, сбой генерации)."""
    a = np.asarray(im, dtype=np.float32)
    return not np.isfinite(a).all() or a.std() < 2.0


class SD:
    """Ленивые модели SD 1.5 (грузятся только если нужны выбранным генераторам)."""

    def __init__(self, gens: list[str]):
        import torch
        from diffusers import (
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionPipeline,
        )

        self.torch = torch
        kw = dict(
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.base = self.i2i = self.inp = None
        if {"vae", "img2img"} & set(gens):
            self.base = StableDiffusionPipeline.from_pretrained(SD_BASE, **kw).to("cuda")
            self.base.set_progress_bar_config(disable=True)
            self.i2i = StableDiffusionImg2ImgPipeline(**self.base.components)
            self.i2i.set_progress_bar_config(disable=True)
        if "inpaint" in gens:
            self.inp = StableDiffusionInpaintPipeline.from_pretrained(SD_INPAINT, **kw).to("cuda")
            self.inp.set_progress_bar_config(disable=True)
        self.hard, _ = face_mask(512)

    def _gens(self, gen: str, stems: list[str]):
        t = self.torch
        return [t.Generator("cuda").manual_seed(seed_of(gen, s)) for s in stems]

    def vae(self, ims: list[Image.Image]) -> list[Image.Image]:
        t = self.torch
        vae = self.base.vae
        x = np.stack([np.asarray(im, dtype=np.float32) / 127.5 - 1.0 for im in ims])
        x = t.from_numpy(x).permute(0, 3, 1, 2).to("cuda", dtype=vae.dtype)
        with t.no_grad():
            z = vae.encode(x).latent_dist.mode()
            y = vae.decode(z).sample
        y = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in y]

    def img2img(self, ims: list[Image.Image], stems: list[str]) -> list[Image.Image]:
        return self.i2i(
            prompt=[PROMPT] * len(ims),
            image=ims,
            strength=I2I_STRENGTH,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            generator=self._gens("img2img", stems),
        ).images

    def inpaint(self, ims: list[Image.Image], stems: list[str]) -> list[Image.Image]:
        out = self.inp(
            prompt=[PROMPT] * len(ims),
            image=ims,
            mask_image=[self.hard] * len(ims),
            height=512,
            width=512,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            generator=self._gens("inpaint", stems),
        ).images
        # вставка фона — в generate(), на 256 после сжатия «как у исходника»;
        # вырожденность проверяем ДО неё: чёрное лицо на живом фоне прошло бы
        return [None if _is_degenerate(g) else g for g in out]


def generate(args) -> None:
    out = Path(args.out)
    reals = select_sources(sorted((Path(args.src) / "real").glob("*.jpg")), args.n)
    fakes = select_sources(sorted((Path(args.src) / "fake").glob("*.jpg")), args.n)
    reals = reals[args.part :: args.nparts]
    fakes = fakes[args.part :: args.nparts]
    gens = list(args.gens)
    print(f"часть {args.part}/{args.nparts}: исходников {len(reals)}, StyleGAN {len(fakes)}")

    for p in fakes if "stylegan" in gens else []:
        dst = out / "stylegan" / p.name
        if not dst.exists():
            _write(dst, finish(up512(Image.open(p))))

    sd_gens = [g for g in gens if g in ("vae", "img2img", "inpaint")]
    sd = SD(sd_gens) if sd_gens else None
    _, soft256 = face_mask(256)
    bad = {g: 0 for g in gens}
    todo = [p for p in reals if any(not (out / g / p.name).exists() for g in gens if g != "stylegan")]
    print(f"осталось исходников: {len(todo)}")
    for i in range(0, len(todo), args.batch):
        batch = todo[i : i + args.batch]
        srcs = [Image.open(p).convert("RGB") for p in batch]
        ups = [up512(x) for x in srcs]
        stems = [p.stem for p in batch]
        if "real" in gens:
            for p, u in zip(batch, ups):
                _write(out / "real" / p.name, finish(u))
        for g in sd_gens:
            need = [k for k, p in enumerate(batch) if not (out / g / p.name).exists()]
            if not need:
                continue
            sub_ims, sub_stems = [ups[k] for k in need], [stems[k] for k in need]
            res = sd.vae(sub_ims) if g == "vae" else getattr(sd, g)(sub_ims, sub_stems)
            for k, im in zip(need, res):
                if im is None or _is_degenerate(im):
                    bad[g] += 1  # не пишем: реальный кадр без пары уйдёт на этапе splits
                    continue
                x = as_source(im)
                if g == "inpaint":
                    # фон — исходные пиксели 140k с их единственным сжатием
                    x = composite(x, srcs[k], soft256)
                _write(out / g / batch[k].name, finish(up512(x)))
        if (i // args.batch) % 25 == 0:
            print(f"  {i + len(batch)}/{len(todo)} | вырожденных: {bad}", flush=True)
    print(f"готово | вырожденных: {bad}")


def build_rows(out: Path) -> pd.DataFrame:
    """Манифест: только полные пары (реальный кадр есть и у каждого SD-генератора
    есть его фейк) + StyleGAN. Неполные пары отбрасываются целиком."""
    have = {g: {p.name for p in (out / g).glob("*.jpg")} for g in GENERATORS}
    paired = have["real"] & have["vae"] & have["img2img"] & have["inpaint"]
    rows = []
    for g in ("real", "vae", "img2img", "inpaint"):
        for name in sorted(paired):
            stem = Path(name).stem
            rows.append(
                {
                    "path": str(out / g / name),
                    "label": int(g != "real"),
                    "source": f"{g}_p",
                    "group": f"ffhq_src::{stem}",
                    "split": split_of(stem),
                }
            )
    for name in sorted(have["stylegan"]):
        stem = Path(name).stem
        rows.append(
            {
                "path": str(out / "stylegan" / name),
                "label": 1,
                "source": "stylegan_p",
                "group": f"stylegan_src::{stem}",
                "split": split_of(stem),
            }
        )
    return pd.DataFrame(rows)


def write_splits(df: pd.DataFrame, splits_root: Path, version: str) -> list[str]:
    """pairs_<g>: реальные + один генератор; pairs_all: реальные + все фейки."""
    written = []
    real = df[df.source == "real_p"]
    variants = {f"pairs_{g}": [f"{g}_p"] for g in GENERATORS if g != "real"}
    variants["pairs_all"] = [f"{g}_p" for g in GENERATORS if g != "real"]
    for name, srcs in variants.items():
        sub = pd.concat([real, df[df.source.isin(srcs)]])
        d = splits_root / name / version
        d.mkdir(parents=True, exist_ok=True)
        for s in ("train", "val", "test"):
            part = sub[sub.split == s][["path", "label", "source", "group"]]
            part.to_csv(d / f"{s}.csv", index=False)
        written.append(name)
    return written


def splits(args) -> None:
    out = Path(args.out)
    df = build_rows(out)
    df.to_csv(out / "manifest.csv", index=False)
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())
    names = write_splits(df, Path(args.splits_root), args.version)
    print(f"сплиты {args.version}: {names}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--src", default=str(SRC_140K), help="папка с real/ и fake/")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=3000)
    g.add_argument("--gens", nargs="+", default=list(GENERATORS), choices=GENERATORS)
    g.add_argument("--part", type=int, default=0)
    g.add_argument("--nparts", type=int, default=1)
    g.add_argument("--batch", type=int, default=8)
    s = sub.add_parser("splits")
    s.add_argument("--out", required=True)
    s.add_argument("--splits-root", default=str(REPO / "data/splits"))
    s.add_argument("--version", default="v1")
    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else splits(args)


if __name__ == "__main__":
    main()
