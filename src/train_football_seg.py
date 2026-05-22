"""Fine-tune YOLOv11n-seg on SAM-generated football player masks."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import SEG_DATASET_YAML
from src.train_common import ROOT, base_train_kwargs


def train(
    batch: int = 8,
    device: str = "mps",
    name: str = "run_v1",
    epochs: int = 100,
    freeze: int = 10,
    data: str | None = None,
) -> None:
    data_yaml = Path(data).resolve() if data else SEG_DATASET_YAML
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run generate_seg_labels.py first."
        )

    model = YOLO("yolo11n-seg.pt")
    kwargs = base_train_kwargs(
        str(data_yaml),
        str(ROOT / "football_tracker_seg"),
        name,
        epochs=epochs,
        batch=batch,
        device=device,
        freeze=freeze,
    )
    kwargs["overlap_mask"] = True
    kwargs["mask_ratio"] = 4
    model.train(**kwargs)
    weights = ROOT / "football_tracker_seg" / name / "weights" / "best.pt"
    print(f"Training complete. Weights: {weights}")
    print("Run: python -m src.validate_seg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="run_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Frozen backbone layers for early epochs (0 = full fine-tune)",
    )
    parser.add_argument("--data", default=None, help="Path to data.yaml (overrides default SEG_DATASET_YAML)")
    args = parser.parse_args()
    train(
        batch=args.batch,
        device=args.device,
        name=args.name,
        epochs=args.epochs,
        freeze=args.freeze,
        data=args.data,
    )
