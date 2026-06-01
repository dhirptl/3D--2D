"""Fine-tune a YOLOv11 segmentation model on football player masks."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import (
    SEG_CLOSE_MOSAIC,
    SEG_DATASET_YAML,
    SEG_MASK_RATIO,
    SEG_TRAIN_COPY_PASTE,
    SEG_TRAIN_DEGREES,
    SEG_TRAIN_ERASING,
    SEG_TRAIN_MODEL,
)
from src.dataset_yaml import validate_dataset_yaml
from src.train_common import DEFAULT_IMGSZ, ROOT, base_train_kwargs

CANONICAL_SEG_RUN = "run_v3"


def train(
    batch: int = 8,
    device: str = "mps",
    name: str = CANONICAL_SEG_RUN,
    epochs: int = 100,
    freeze: int = 10,
    data: str | None = None,
    model_name: str = SEG_TRAIN_MODEL,
    imgsz: int = DEFAULT_IMGSZ,
    mask_ratio: int = SEG_MASK_RATIO,
    degrees: float = SEG_TRAIN_DEGREES,
    erasing: float = SEG_TRAIN_ERASING,
    copy_paste: float = SEG_TRAIN_COPY_PASTE,
    close_mosaic: int = SEG_CLOSE_MOSAIC,
) -> None:
    data_yaml = Path(data).resolve() if data else SEG_DATASET_YAML
    validate_dataset_yaml(data_yaml, task="seg")

    model = YOLO(model_name)
    kwargs = base_train_kwargs(
        str(data_yaml),
        str(ROOT / "football_tracker_seg"),
        name,
        epochs=epochs,
        batch=batch,
        device=device,
        freeze=freeze,
    )
    kwargs["imgsz"] = int(imgsz)
    kwargs["overlap_mask"] = True
    kwargs["mask_ratio"] = int(mask_ratio)
    kwargs["degrees"] = float(degrees)
    kwargs["erasing"] = float(erasing)
    kwargs["copy_paste"] = float(copy_paste)
    kwargs["close_mosaic"] = int(close_mosaic)
    model.train(**kwargs)
    weights = ROOT / "football_tracker_seg" / name / "weights" / "best.pt"
    print(f"Training complete. Weights: {weights}")
    print(f"Inference default: SEG_MODEL_PATH or --model {weights}")
    print("Run: python -m src.validate_seg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default=CANONICAL_SEG_RUN)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Frozen backbone layers for early epochs (0 = full fine-tune)",
    )
    parser.add_argument("--model-name", default=SEG_TRAIN_MODEL, help="Base Ultralytics seg model")
    parser.add_argument("--data", default=None, help="Path to data.yaml (overrides default SEG_DATASET_YAML)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--mask-ratio", type=int, default=SEG_MASK_RATIO)
    parser.add_argument("--degrees", type=float, default=SEG_TRAIN_DEGREES)
    parser.add_argument("--erasing", type=float, default=SEG_TRAIN_ERASING)
    parser.add_argument("--copy-paste", type=float, default=SEG_TRAIN_COPY_PASTE)
    parser.add_argument("--close-mosaic", type=int, default=SEG_CLOSE_MOSAIC)
    args = parser.parse_args()
    train(
        batch=args.batch,
        device=args.device,
        name=args.name,
        epochs=args.epochs,
        freeze=args.freeze,
        data=args.data,
        model_name=args.model_name,
        imgsz=args.imgsz,
        mask_ratio=args.mask_ratio,
        degrees=args.degrees,
        erasing=args.erasing,
        copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic,
    )
