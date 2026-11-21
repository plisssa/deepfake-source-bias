"""Какая схема хвоста уравнивает историю сжатия у реальных и SD-фейков.

В v1 одна «блочность» отделяла SD-фейки с AUC 0.94. Синтетика не
годится для выбора исправления: мера блочности чувствительна к гладкости
содержимого. Здесь — настоящий генератор (только VAE SD 1.5, на CPU) на N
тестовых исходниках и несколько схем подготовки; для каждой — ранговый AUC
блочности и веса файла «реальное против фейка». Нужна схема, где оба ≈ 0.5.

    python -m scripts.pairs_history_check --n 300
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from scripts import build_pairs as bp
from scripts.run_pairs_matrix import blockiness


def jpeg444(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=95, subsampling=0)
    return b.getvalue()


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def box(im512: Image.Image) -> Image.Image:
    return im512.convert("RGB").resize((256, 256), Image.BOX)


# схема → (реальный кадр, фейк) из исходника src (256, одно сжатие 140k) и выхода G (512)
SCHEMES = {
    "v1_jpeg_tail": lambda src, g: (bp.finish(bp.up512(src)), bp.finish(g)),
    "v2_src_hist": lambda src, g: (bp.finish(bp.up512(src)), bp.finish(bp.up512(bp.as_source(g)))),
    "png_tail": lambda src, g: (png(box(bp.up512(src))), png(box(g))),
    "png_tail_src_hist": lambda src, g: (png(box(bp.up512(src))), png(bp.as_source(g))),
    # одинаковая история у всех пикселей (одно сжатие q95 4:2:0 + общий цикл
    # ресайза) и хранение без потерь — единственная схема, симметричная по построению
    "v2_png_tail": lambda src, g: (png(box(bp.up512(src))), png(box(bp.up512(bp.as_source(g))))),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    import torch
    from diffusers import AutoencoderKL

    torch.set_num_threads(8)
    vae = AutoencoderKL.from_pretrained(bp.SD_BASE, subfolder="vae", variant="fp16").float().eval()
    test = pd.read_csv(bp.REPO / "data/splits/pairs_vae/v1/test.csv")
    names = [p.split("/")[-1] for p in test[test.label == 0].path][: args.n]
    feats = {k: {"real": [], "fake": []} for k in SCHEMES}
    for i, name in enumerate(names):
        src = Image.open(bp.SRC_140K / "real" / name).convert("RGB")
        x = torch.from_numpy(np.asarray(bp.up512(src), np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        with torch.no_grad():
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
        g = Image.fromarray(((y[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy())
        for k, f in SCHEMES.items():
            r, fk = f(src, g)
            for cls, data in (("real", r), ("fake", fk)):
                feats[k][cls].append((blockiness(io.BytesIO(data)), len(data)))
        if i % 50 == 0:
            print(f"{i}/{len(names)}", flush=True)
    rows = []
    for k, d in feats.items():
        r, f = np.array(d["real"]), np.array(d["fake"])
        y = np.r_[np.zeros(len(r)), np.ones(len(f))]
        rows.append(
            {
                "scheme": k,
                "auc_blockiness": roc_auc_score(y, np.r_[r[:, 0], f[:, 0]]),
                "auc_filesize": roc_auc_score(y, np.r_[r[:, 1], f[:, 1]]),
                "block_real": np.median(r[:, 0]),
                "block_fake": np.median(f[:, 0]),
            }
        )
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(bp.REPO / "reports/tables/pairs_history_check.csv", index=False)


if __name__ == "__main__":
    main()
