"""Training entry point.

Usage:
    python train.py [--config configs/train.yaml]

Reads configs/train.yaml and passes values directly to ultralytics YOLO.train().
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

_TRAIN_PARAMS = {
    "data", "batch", "device", "name", "epochs", "freeze",
    "imgsz", "mosaic", "mixup", "fliplr", "hsv_h", "hsv_s",
    "scale", "translate", "patience", "project",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train detection model from configs/train.yaml")
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()

    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open() as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if v is not None})
    else:
        print(f"[train] Config not found at {config_path}, using ultralytics defaults")

    model_name = cfg.pop("model_name", "yolo11m.pt")
    kwargs = {k: v for k, v in cfg.items() if k in _TRAIN_PARAMS}
    print(f"[train] model={model_name}  params={kwargs}")

    model = YOLO(model_name)
    model.train(**kwargs)


if __name__ == "__main__":
    main()
