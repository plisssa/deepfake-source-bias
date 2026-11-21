"""Автоскачивание датасетов исследования (задача «одна команда на сервере»).

Реестр DATASETS описывает источники: kagglehub / huggingface / прямой URL.
Скачивание параллельное (IO-bound → потоки), с манифестом и симлинками в
data/raw/<name>, чтобы конфиги data/*.yaml могли ссылаться на стабильные пути.

Запуск:
    python -m scripts.download_data                      # всё из реестра
    python -m scripts.download_data --only ffhq,dff      # выборочно
    python -m scripts.download_data --list               # показать реестр

Токены: Kaggle — ~/.kaggle/kaggle.json или KAGGLE_USERNAME/KAGGLE_KEY;
HuggingFace — HF_TOKEN (нужен только для gated-датасетов).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from deepfakes.utils.env import load_project_dotenv
from deepfakes.utils.logging import configure_logging, get_logger

load_project_dotenv()
log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST = REPO_ROOT / "data" / "downloads.json"


@dataclass
class DatasetSpec:
    name: str  # короткое имя → data/raw/<name>
    kind: str  # 'real' | 'fake'
    family: str  # 'real' | 'gan' | 'diffusion' | 'mixed'
    method: str  # 'kagglehub' | 'huggingface' | 'url'
    ref: str  # kaggle ref / HF repo_id / URL
    note: str = ""
    sha256: str | None = None  # для method='url'
    subdir: str | None = None  # какая поддиректория внутри скачанного нужна
    extra: dict = field(default_factory=dict)


# Download helpers for supported image corpora.
DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="real_fake_140k",
        kind="fake",
        family="gan",
        method="kagglehub",
        ref="xhlulu/140k-real-and-fake-faces",
        note="70k real (Flickr) + 70k StyleGAN fake, 256x256 — базовый GAN-домен",
    ),
    DatasetSpec(
        name="celeba",
        kind="real",
        family="real",
        method="kagglehub",
        ref="jessicali9530/celeba-dataset",
        note="202k реальных лиц знаменитостей (таблица №2)",
    ),
    DatasetSpec(
        name="lfw",
        kind="real",
        family="real",
        method="kagglehub",
        ref="jessicali9530/lfw-dataset",
        note="13k реальных лиц in-the-wild (таблица №1)",
    ),
    DatasetSpec(
        name="dff",
        kind="fake",
        family="diffusion",
        method="huggingface",
        ref="OpenRL/DeepFakeFace",
        note="DeepFakeFace: 120k diffusion-фейков (Stable Diffusion и др.), таблица №16",
    ),
    DatasetSpec(
        name="ifakefacedb",
        kind="fake",
        family="gan",
        method="url",
        ref="https://github.com/socialabubi/iFakeFaceDB/releases/download/v1.0/iFakeFaceDB.zip",
        note="87k StyleGAN-фейков (таблица №6); при 404 скачать вручную со страницы проекта",
    ),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_into_raw(name: str, target: Path) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    link = RAW_DIR / name
    if link.is_symlink() or link.exists():
        return link
    link.symlink_to(target, target_is_directory=target.is_dir())
    return link


def _download_kagglehub(spec: DatasetSpec) -> Path:
    import kagglehub

    path = Path(kagglehub.dataset_download(spec.ref))
    return path if spec.subdir is None else path / spec.subdir


def _download_huggingface(spec: DatasetSpec) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=spec.ref, repo_type="dataset", **spec.extra))
    return path if spec.subdir is None else path / spec.subdir


def _download_url(spec: DatasetSpec) -> Path:
    import urllib.request
    import zipfile

    out_dir = RAW_DIR / f"_url_{spec.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / Path(spec.ref).name
    if not fname.exists():
        log.info(f"[{spec.name}] downloading {spec.ref}")
        urllib.request.urlretrieve(spec.ref, fname)
    if spec.sha256:
        actual = _sha256(fname)
        if actual != spec.sha256:
            raise ValueError(f"[{spec.name}] sha256 mismatch: {actual} != {spec.sha256}")
    if fname.suffix == ".zip":
        marker = out_dir / ".extracted"
        if not marker.exists():
            log.info(f"[{spec.name}] extracting {fname.name}")
            with zipfile.ZipFile(fname) as zf:
                zf.extractall(out_dir)
            marker.touch()
    return out_dir if spec.subdir is None else out_dir / spec.subdir


def download_one(spec: DatasetSpec) -> dict:
    try:
        if spec.method == "kagglehub":
            path = _download_kagglehub(spec)
        elif spec.method == "huggingface":
            path = _download_huggingface(spec)
        elif spec.method == "url":
            path = _download_url(spec)
        else:
            raise ValueError(f"Unknown method: {spec.method}")
        link = _link_into_raw(spec.name, path)
        n_files = sum(1 for _ in path.rglob("*") if _.is_file()) if path.is_dir() else 1
        log.info(f"[{spec.name}] OK → {link} ({n_files} files)")
        return {
            **asdict(spec),
            "status": "ok",
            "path": str(path),
            "link": str(link),
            "n_files": n_files,
        }
    except Exception as e:
        log.error(f"[{spec.name}] FAILED: {e}")
        return {**asdict(spec), "status": "failed", "error": str(e)}


def main() -> None:
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="имена через запятую (см. --list)")
    ap.add_argument("--list", action="store_true", help="показать реестр и выйти")
    ap.add_argument("--jobs", type=int, default=3, help="параллельных загрузок (default 3)")
    args = ap.parse_args()

    if args.list:
        for s in DATASETS:
            print(f"{s.name:18s} {s.kind:5s} {s.family:9s} {s.method:12s} {s.ref}  — {s.note}")
        return

    specs = DATASETS
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        unknown = wanted - {s.name for s in DATASETS}
        if unknown:
            sys.exit(f"Unknown dataset names: {sorted(unknown)}. See --list")
        specs = [s for s in DATASETS if s.name in wanted]

    log.info(f"Downloading {len(specs)} dataset(s) with {args.jobs} parallel workers")
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(download_one, s): s for s in specs}
        for fut in as_completed(futures):
            results.append(fut.result())

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for r in results:
        existing[r["name"]] = r
    MANIFEST.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    failed = [r["name"] for r in results if r["status"] != "ok"]
    ok = [r["name"] for r in results if r["status"] == "ok"]
    log.info(f"Done: ok={ok}, failed={failed}. Manifest: {MANIFEST}")
    if ok:
        log.info(
            "Пропишите пути в .env, например: DATA_REAL_DIR=data/raw/celeba "
            "DATA_FAKE_DIR=data/raw/real_fake_140k (или укажите data/raw/<name> в configs/data/*.yaml)"
        )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
