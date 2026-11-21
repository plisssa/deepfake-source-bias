"""Скачать веса timm заранее, на login-узле (compute-узлы офлайн).

Запускать ТОЛЬКО как модуль — `python -m scripts.warm_weights ...`: тогда при
импорте читается `.env` проекта, и веса ложатся в тот HF_HOME, из которого их
потом прочитает задача. 27.09.2026 прогрев через `python -c` не видел `.env`,
веса ушли в ~/.cache/huggingface, и GPU-прогон с HF_HUB_OFFLINE=1 упал с
`extraction failed for ALL 28000 images`.

    python -m scripts.warm_weights vit_large_patch14_clip_224.openai resnet50.a1_in1k

Префикс `hf:` — целый репозиторий diffusers (для build_pairs): только fp16-веса
и конфиги, без safety checker (608 МБ, в пайплайне отключён).

    python -m scripts.warm_weights hf:stable-diffusion-v1-5/stable-diffusion-v1-5
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from deepfakes.utils.env import load_project_dotenv

load_project_dotenv()


def warm_diffusers(repo: str) -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo,
        allow_patterns=["*.json", "*.txt", "*.fp16.safetensors"],
        ignore_patterns=["safety_checker/*.safetensors"],
    )
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    print(f"ok {repo}: {path} ({size:.1f} ГБ)")


def main(names: list[str]) -> None:
    print(f"HF_HOME = {os.environ.get('HF_HOME', '(не задан — будет ~/.cache/huggingface)')}")
    for n in [n for n in names if n.startswith("hf:")]:
        warm_diffusers(n[3:])
    names = [n for n in names if not n.startswith("hf:")]
    if names:
        import timm
    for n in names:
        m = timm.create_model(n, pretrained=True, num_classes=0)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        cfg = timm.data.resolve_model_data_config(m)
        print(f"ok {n}: {n_params:.0f}M параметров, mean={cfg['mean']}, std={cfg['std']}")
    print("WEIGHTS_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
