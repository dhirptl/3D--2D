"""Root-level training wrapper for the optimization loop.

Usage:
    python train.py [--config configs/train.yaml]

Reads configs/train.yaml and passes values to src.train_football_seg.train().
Unknown YAML keys are silently ignored, so you can add notes/comments freely.
"""

import argparse
from pathlib import Path

import yaml

from src.train_football_seg import train

_TRAIN_PARAMS = {
    "batch", "device", "name", "epochs", "freeze",
    "model_name", "imgsz", "mask_ratio", "degrees",
    "erasing", "copy_paste", "close_mosaic",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train segmentation model from configs/train.yaml")
    ap.add_argument("--config", default="configs/train.yaml", help="Path to train YAML config")
    args = ap.parse_args()

    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open() as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if v is not None})
    else:
        print(f"[train] Config not found at {config_path}, using src/train_football_seg.py defaults")

    kwargs = {k: v for k, v in cfg.items() if k in _TRAIN_PARAMS}
    print(f"[train] Training with: {kwargs}")
    train(**kwargs)


if __name__ == "__main__":
    main()
