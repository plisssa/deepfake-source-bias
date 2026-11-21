"""Генерация и сохранение фиксированных сплитов train/val/test.

Запуск:
    python -m scripts.prepare_splits data=default cv=holdout seed=42

Создаёт data/splits/<data.name>/<splits_version>/{train,val,test}.csv
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from deepfakes.io.collect import DatasetBundle, DatasetSource
from deepfakes.io.splits import SplitConfig, make_holdout_split, save_split_csv
from deepfakes.utils.logging import configure_logging, get_logger
from deepfakes.utils.seed import seed_everything


def _bundle_from_cfg(data_cfg: DictConfig) -> DatasetBundle:
    def _mk(src_cfg, label: int) -> DatasetSource:
        d = OmegaConf.to_container(src_cfg, resolve=True)
        d.setdefault("label", label)
        return DatasetSource(**d)

    real = [_mk(s, 0) for s in data_cfg.real_sources]
    fake = [_mk(s, 1) for s in data_cfg.fake_sources]
    return DatasetBundle(real_sources=real, fake_sources=fake)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    builder = cfg.data.get("splits_builder")
    if builder:
        # парные наборы: сплит по исходному кадру пишет свой скрипт; перемешивание
        # здесь развело бы реальный кадр и его фейки по разные стороны
        raise SystemExit(f"сплиты {cfg.data.name} строит scripts/{builder}.py, не prepare_splits")
    seed_everything(cfg.seed)
    log.info("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    bundle = _bundle_from_cfg(cfg.data)
    X, y, sources, groups = bundle.collect(seed=cfg.seed)

    sp_cfg = SplitConfig(
        test_size=cfg.cv.test_size,
        val_size=getattr(cfg.cv, "val_size", 0.15),
        seed=cfg.seed,
        stratify=cfg.cv.stratify,
    )
    splits = make_holdout_split(X, y, sources, sp_cfg, groups=groups)

    out_dir = Path(cfg.paths.splits_dir) / cfg.data.name / cfg.data.splits_version
    save_split_csv(splits, out_dir)
    log.info(f"✔ splits saved to: {out_dir}")


if __name__ == "__main__":
    main()
