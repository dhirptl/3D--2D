"""Shared Ultralytics training hyperparameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EPOCHS = 150
DEFAULT_PATIENCE = 40
DEFAULT_FREEZE = 10
DEFAULT_BATCH = 2
DEFAULT_IMGSZ = 1280
DEFAULT_DEVICE = "mps"


def base_train_kwargs(
    data_yaml: str,
    project: str,
    name: str,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH,
    device: str = DEFAULT_DEVICE,
    freeze: int = DEFAULT_FREEZE,
    patience: int = DEFAULT_PATIENCE,
) -> dict:
    return {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": DEFAULT_IMGSZ,
        "batch": batch,
        "patience": patience,
        "freeze": freeze,
        "mosaic": 1.0,
        "mixup": 0.15,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "scale": 0.7,
        "translate": 0.1,
        "device": device,
        "project": project,
        "name": name,
    }
